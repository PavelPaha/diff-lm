"""Fast, deterministic HellaSwag quality gate for locally modified LLaDA.

This intentionally avoids lm-evaluation-harness and datasets.  It uses the
same masked-diffusion conditional-likelihood estimator as LLaDA's official
`eval_llada.py`, on a fixed 1,024-example HellaSwag validation subset.  Run it
with two GPUs:

    torchrun --standalone --nproc-per-node=2 -m dlm_attn_res.evaluation.hellaswag_subset
"""

import argparse
import json
import os
import random
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..runtime.storage import DATASETS_HOME, configure_storage

configure_storage()

from transformers import AutoTokenizer

from ..models.llada import LLaDAConfig, LLaDAModelLM


DEFAULT_DATA = DATASETS_HOME / "hellaswag_val.jsonl"
DATA_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def load_documents(path: Path, subset_size: int, seed: int):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    with path.open() as handle:
        documents = [json.loads(line) for line in handle]
    selected = random.Random(seed).sample(documents, subset_size)
    return selected


def format_context(document):
    # Same text normalization as lm-eval's HellaSwag task.
    return document["ctx_a"] + " " + document["ctx_b"].capitalize()


def encode_pair(tokenizer, context, continuation):
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    prefix = tokenizer(context)["input_ids"]
    whole = tokenizer(context + continuation)["input_ids"]
    return prefix, whole[len(prefix) :]


@torch.no_grad()
def masked_log_likelihood(model, prefix, target, mask_id, batch_size, mc_num):
    sequence = torch.tensor(prefix + target, dtype=torch.long, device=model.device)
    sequence = sequence.unsqueeze(0).repeat(batch_size, 1)
    prompt_index = torch.arange(sequence.shape[1], device=model.device) < len(prefix)
    target_length = len(target)
    losses = []
    for _ in range(mc_num // batch_size):
        k = torch.randint(1, target_length + 1, (), device=model.device)
        counts = torch.round(
            torch.linspace(float(k), k + (batch_size - 1) * (target_length / batch_size),
                           steps=batch_size, device=model.device)
        ).long()
        counts = ((counts - 1) % target_length) + 1
        indices = torch.arange(target_length, device=model.device).repeat(batch_size, 1)
        masked = indices < counts.unsqueeze(1)
        for row in range(batch_size):
            masked[row] = masked[row][torch.randperm(target_length, device=model.device)]
        masked = torch.cat(
            [torch.zeros(batch_size, len(prefix), dtype=torch.bool, device=model.device), masked], dim=1
        )
        noisy = torch.where(masked, mask_id, sequence)
        logits = model(noisy).logits
        token_loss = F.cross_entropy(logits[masked], sequence[masked], reduction="none")
        mask_probability = (counts / target_length).unsqueeze(1).expand_as(sequence)
        losses.append((token_loss / mask_probability[masked]).sum().div(batch_size).item())
    return -sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--subset-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mc-num", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/hellaswag_1024_mc128_local.json"))
    args = parser.parse_args()
    assert args.mc_num % args.batch_size == 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    config = LLaDAConfig.from_pretrained(args.model_path)
    model = LLaDAModelLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    documents = load_documents(DEFAULT_DATA, args.subset_size, args.seed)[rank::world_size]

    correct = 0
    for index, document in enumerate(documents, start=1):
        context = format_context(document)
        scores = []
        for ending in document["endings"]:
            prefix, target = encode_pair(tokenizer, context, " " + ending)
            # HellaSwag's reported metric is acc_norm: compare the average
            # token likelihood, not the total likelihood.  Without this,
            # short continuations receive an artificial advantage and the
            # value is not comparable to the LLaDA paper.
            score = masked_log_likelihood(
                model, prefix, target, 126336, args.batch_size, args.mc_num
            )
            scores.append(score / len(target))
        correct += int(max(range(4), key=scores.__getitem__) == int(document["label"]))
        if index % 32 == 0:
            print(f"rank={rank} {index}/{len(documents)} accuracy={correct / index:.4f}", flush=True)

    totals = torch.tensor([correct, len(documents)], dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "task": "hellaswag",
            "subset_size": int(totals[1]),
            "subset_seed": args.seed,
            "mc_num": args.mc_num,
            "batch_size": args.batch_size,
            "architecture": "local LLaDAModelLM",
            "model_path": args.model_path,
            "metric": "acc_norm",
            "accuracy": totals[0].item() / totals[1].item(),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
