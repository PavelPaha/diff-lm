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
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoTokenizer
import wandb
from tqdm.auto import tqdm

from dlm_attn_res.models.llada import LLaDAConfig, LLaDAModelLM
from dlm_attn_res.models.llada.configuration import ActivationCheckpointingStrategy


def norm(parameters):
    values = [p.grad.detach().float().norm() ** 2 for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt().item() if values else 0.0


def attention_residual_image(source_maps, num_layers):
    """Render depth-wise AR routing as an explicit RGB heatmap.

    Row `i` is the residual routing used before transformer block `i`; column
    `j` is the source representation from depth `j`. Colors show the weight
    *relative to uniform routing* in that row: grey = uniform, red = preferred
    source, blue = suppressed source. This makes small weights visible and
    avoids W&B/Pillow treating NaNs outside the triangle as black pixels.
    """
    routing = np.zeros((num_layers, num_layers), dtype=np.float32)
    valid = np.zeros((num_layers, num_layers), dtype=bool)
    for block_idx, weights in enumerate(source_maps):
        count = weights.numel()
        # `weights * count - 1` is zero for an exactly uniform distribution.
        routing[block_idx, :count] = weights.numpy() * count - 1.0
        valid[block_idx, :count] = True

    max_deviation = max(float(np.abs(routing[valid]).max(initial=0.0)), 1e-6)
    normalized = np.clip(routing / max_deviation, -1.0, 1.0)
    rgb = np.full((num_layers, num_layers, 3), 42, dtype=np.uint8)
    # A compact diverging palette: blue -> neutral grey -> red.
    positive = normalized >= 0
    rgb[..., 0][valid] = 180
    rgb[..., 1][valid] = 180
    rgb[..., 2][valid] = 180
    rgb[..., 0][valid & positive] = 180 + (75 * normalized[valid & positive]).astype(np.uint8)
    rgb[..., 1][valid & positive] = 180 - (120 * normalized[valid & positive]).astype(np.uint8)
    rgb[..., 2][valid & positive] = 180 - (120 * normalized[valid & positive]).astype(np.uint8)
    negative = valid & ~positive
    magnitude = -normalized[negative]
    rgb[..., 0][negative] = 180 - (120 * magnitude).astype(np.uint8)
    rgb[..., 1][negative] = 180 - (120 * magnitude).astype(np.uint8)
    rgb[..., 2][negative] = 180 + (75 * magnitude).astype(np.uint8)
    return wandb.Image(
        rgb,
        caption="Attention Residual routing: row=target block, column=source depth; grey=uniform, red=preferred, blue=suppressed",
    )


def packed_token_batch(iterator, tokenizer, token_buffer, batch_size, sequence_length):
    """Return exact-length batches by concatenating FineWeb documents with EOS.

    This is necessary for the configured global token batch to mean what it
    says: truncating one document per micro-step otherwise produces a highly
    variable, usually much shorter sequence.
    """
    tokens_needed = batch_size * sequence_length
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The LLaDA tokenizer must define eos_token_id for packed pre-training data")
    while len(token_buffer) < tokens_needed:
        text = next(iterator)["text"]
        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        token_buffer.append(eos_token_id)
    batch_tokens = token_buffer[:tokens_needed]
    del token_buffer[:tokens_needed]
    return torch.tensor(batch_tokens, dtype=torch.long).view(batch_size, sequence_length)


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
    diffusion_cfg = cfg.get("diffusion", {})
    masking_epsilon = diffusion_cfg.get("masking_epsilon", 1e-3)
    random_length_probability = diffusion_cfg.get("random_length_probability", 0.01)
    if not 0.0 < masking_epsilon < 1.0:
        raise ValueError("diffusion.masking_epsilon must be in (0, 1)")
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
    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
        # Reuse DDP's all-reduce buckets as gradient storage after the first
        # iteration instead of allocating a second full gradient buffer.
        gradient_as_bucket_view=True,
    )

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
    optimizer_kwargs = {
        "betas": (opt_cfg["adam_beta1"], opt_cfg["adam_beta2"]),
        "eps": opt_cfg["adam_eps"],
    }
    if opt_cfg.get("zero_stage_1", False):
        # DDP replicates model parameters, but this shards Adam's m/v state
        # across ranks. For an 8B model that saves roughly 32 GiB per H200.
        optimizer = ZeroRedundancyOptimizer(
            parameter_groups,
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    tokens_per_step = world * opt_cfg["micro_batch_size"] * model_cfg["context_length"] * opt_cfg["gradient_accumulation_steps"]
    total_steps = math.ceil(data_cfg["target_tokens"] / tokens_per_step)
    warmup_steps = opt_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = max(1, round(total_steps * opt_cfg["warmup_ratio"]))
    else:
        warmup_steps = int(warmup_steps)
        if not 0 < warmup_steps < total_steps:
            raise ValueError(f"warmup_steps must be in [1, {total_steps - 1}], got {warmup_steps}")
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
    token_buffer = []
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
        mask_probability_sum = 0.0
        masked_fraction_sum = 0.0
        for micro in range(opt_cfg["gradient_accumulation_steps"]):
            ids = packed_token_batch(
                iterator,
                tokenizer,
                token_buffer,
                opt_cfg["micro_batch_size"],
                context,
            ).to(device)
            # LLaDA uses a 4096-token block normally, and shortens 1% of
            # blocks to a uniformly sampled length for length robustness.
            if torch.rand((), device=device) < random_length_probability:
                random_length = torch.randint(1, ids.shape[1] + 1, (), device=device).item()
                ids = ids[:, :random_length]

            # Sample one diffusion time per sequence, not one time for the
            # whole batch. The epsilon prevents an exactly zero mask rate.
            batch_size, sequence_length = ids.shape
            p_mask = (1.0 - masking_epsilon) * torch.rand(batch_size, device=device) + masking_epsilon
            masked = torch.rand((batch_size, sequence_length), device=device) < p_mask[:, None]
            # With micro-batch one, a very small p_mask can occasionally
            # produce no masked token. Make that edge case trainable instead
            # of passing an empty tensor to cross_entropy.
            for row in range(batch_size):
                if not masked[row].any():
                    masked[row, torch.randint(sequence_length, (), device=device)] = True
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
                token_loss = F.cross_entropy(logits[masked], ids[masked], reduction="none")
                token_loss = token_loss / p_mask[:, None].expand_as(ids)[masked]
                # This is the LLaDA pre-training objective: normalize by all
                # positions, rather than the variable number of masked ones.
                loss = token_loss.sum() / (batch_size * sequence_length)
                loss = loss / opt_cfg["gradient_accumulation_steps"]
            if loss.requires_grad:
                loss.backward()
            loss_sum += loss.detach().item()
            mask_probability_sum += p_mask.mean().item()
            masked_fraction_sum += masked.float().mean().item()
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
                "train/mask_probability": mask_probability_sum / opt_cfg["gradient_accumulation_steps"],
                "train/masked_fraction": masked_fraction_sum / opt_cfg["gradient_accumulation_steps"],
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
