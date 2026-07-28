import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
import wandb
from tqdm.auto import tqdm

from dlm_attn_res.models.llada import LLaDAConfig, LLaDAModelLM
from dlm_attn_res.models.llada.configuration import ActivationCheckpointingStrategy


def norm(parameters):
    values = [p.grad.detach().float().norm() ** 2 for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt().item() if values else 0.0


def attention_residual_image(source_maps, num_layers):
    """Turn depth-wise AR weights into one triangular W&B heatmap.

    Row `i` is the residual routing used before transformer block `i`; column
    `j` is the source representation from depth `j`.  Each value is averaged
    over the current microbatch and token positions.
    """
    image = np.full((num_layers, num_layers), np.nan, dtype=np.float32)
    for block_idx, weights in enumerate(source_maps):
        image[block_idx, : weights.numel()] = weights.numpy()
    # W&B normalizes the finite values and renders this as a heatmap.
    return wandb.Image(image, caption="Attention Residuals: target block (row) × source depth (column)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = cfg["seed"] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_cfg = cfg["model"]
    opt_cfg, data_cfg = cfg["optimization"], cfg["data"]
    checkpoint = model_cfg["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    config = LLaDAConfig.from_pretrained(checkpoint, local_files_only=True)
    model = LLaDAModelLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.bfloat16).to(device)
    if opt_cfg["activation_checkpointing"]:
        model.model.set_activation_checkpointing(
            ActivationCheckpointingStrategy(opt_cfg["activation_checkpointing"])
        )
    def is_attention_residual_parameter(name):
        # `norm` is the RMSNorm inserted immediately after the AR operator;
        # original LLaDA norms are named `attn_norm` / `ff_norm` and do not
        # match this dotted component.
        return "attn_res" in name or ".norm." in name

    if not opt_cfg["train_base_model"]:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(is_attention_residual_parameter(name))
    model.train()
    # At scale=0 the AR branch is intentionally absent from the graph, so its
    # parameters are temporarily unused.
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)

    attn_params = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and is_attention_residual_parameter(name)
    ]
    base_params = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not is_attention_residual_parameter(name)
    ]
    parameter_groups = []
    if base_params:
        parameter_groups.append({
            "params": base_params,
            "lr": opt_cfg["base_learning_rate"],
            "weight_decay": opt_cfg["weight_decay"],
            "name": "base",
        })
    if attn_params:
        parameter_groups.append({
            "params": attn_params,
            "lr": opt_cfg["attention_residual_learning_rate"],
            "weight_decay": opt_cfg["weight_decay"],
            "name": "attention_residuals",
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(opt_cfg["adam_beta1"], opt_cfg["adam_beta2"]),
        eps=opt_cfg["adam_eps"],
    )
    tokens_per_step = world * opt_cfg["micro_batch_size"] * model_cfg["context_length"] * opt_cfg["gradient_accumulation_steps"]
    total_steps = math.ceil(data_cfg["target_tokens"] / tokens_per_step)
    warmup_steps = max(1, round(total_steps * opt_cfg["warmup_ratio"]))
    def factor(step):
        if step < warmup_steps: return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return opt_cfg["min_lr_ratio"] + (1 - opt_cfg["min_lr_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    dataset = load_dataset(
        data_cfg["dataset"],
        name=data_cfg.get("subset"),
        split=data_cfg["split"],
        streaming=True,
    ).shard(world, rank)
    iterator = iter(dataset)
    if rank == 0:
        run = wandb.init(entity=cfg["logging"]["wandb_entity"], project=cfg["logging"]["wandb_project"], name=cfg["run_name"], config=cfg, settings=wandb.Settings(base_url=cfg["logging"]["wandb_base_url"]))
        # Keep the exact input file in the run, not only its parsed key/value
        # representation shown in the W&B Config panel.
        run.save(str(args.config), base_path=str(args.config.parent), policy="now")
    context, mask_id, processed = model_cfg["context_length"], model_cfg["mask_token_id"], 0
    attention_map_every = cfg["logging"].get("attention_maps_every_steps", 20)
    progress_bar = tqdm(
        range(total_steps),
        disable=rank != 0,
        dynamic_ncols=True,
        desc="FineWeb fine-tune",
    )
    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(opt_cfg["gradient_accumulation_steps"]):
            text = next(iterator)["text"]
            ids = tokenizer(text, truncation=True, max_length=context, return_tensors="pt")["input_ids"]
            if ids.shape[1] < 2: continue
            ids = ids.to(device)
            rate = torch.rand((), device=device)
            masked = torch.rand_like(ids.float()) < rate
            masked[:, 0] = False
            if not masked.any(): masked[0, -1] = True
            corrupted = ids.masked_fill(masked, mask_id)
            ar_scale = min(step / cfg["attention_residuals"]["warmup_steps"], 1.0)
            capture_attention_maps = (
                rank == 0
                and cfg["attention_residuals"]["enabled"]
                and ar_scale > 0.0
                and step % attention_map_every == 0
                and micro == opt_cfg["gradient_accumulation_steps"] - 1
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=corrupted,
                    use_attention_residuals=cfg["attention_residuals"]["enabled"],
                    attention_residual_scale=ar_scale,
                    capture_attention_residual_maps=capture_attention_maps,
                ).logits
                loss = F.cross_entropy(logits[masked], ids[masked]) / opt_cfg["gradient_accumulation_steps"]
            if loss.requires_grad:
                loss.backward()
            loss_sum += loss.detach().item()
            processed += ids.numel() * world
        attn_grad, base_grad = norm(attn_params), norm(base_params)
        trainable_params = base_params + attn_params
        if any(parameter.grad is not None for parameter in trainable_params):
            torch.nn.utils.clip_grad_norm_(trainable_params, opt_cfg["max_grad_norm"])
            optimizer.step()
        scheduler.step()
        if rank == 0 and step % cfg["logging"]["log_every_steps"] == 0:
            metrics = {
                "train/loss": loss_sum,
                "train/tokens": processed,
                "train/lr_base": next((group["lr"] for group in optimizer.param_groups if group["name"] == "base"), 0.0),
                "train/lr_attention_residuals": next((group["lr"] for group in optimizer.param_groups if group["name"] == "attention_residuals"), 0.0),
                "attn_res/scale": ar_scale,
                "grad_norm/attn_res": attn_grad,
                "grad_norm/base": base_grad,
                "system/max_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            if capture_attention_maps:
                metrics["attn_res/source_attention"] = attention_residual_image(
                    model.module.model.last_attention_residual_maps,
                    model.module.config.n_layers,
                )
            run.log(metrics, step=step)
            progress_bar.set_postfix(loss=f"{loss_sum:.4f}", lr=f"{metrics['train/lr_base']:.2e}/{metrics['train/lr_attention_residuals']:.2e}", ar=f"{ar_scale:.3f}")
            if step % 10 == 0:
                tqdm.write(
                    f"step={step}/{total_steps} tokens={processed:,} loss={loss_sum:.4f} "
                    f"lr_base={metrics['train/lr_base']:.3e} "
                    f"lr_ar={metrics['train/lr_attention_residuals']:.3e} ar_scale={ar_scale:.4f}"
                )
    if rank == 0: run.finish()
    dist.destroy_process_group()

if __name__ == "__main__": main()
