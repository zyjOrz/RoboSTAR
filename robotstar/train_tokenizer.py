from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TokenizerConfig
from .data import DistributedLengthBucketSampler, PreparedMotionDataset, collate_motion
from .tokenizer import RobotSTARTokenizer
from .training import autocast_context, is_main, load_yaml_config, save_checkpoint, setup_distributed, unwrap, write_json


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(prediction.shape[1], device=prediction.device).unsqueeze(0) < lengths.unsqueeze(1)
    valid = valid.unsqueeze(-1).to(prediction.dtype)
    return F.smooth_l1_loss(prediction * valid, target * valid)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: str, commit_weight: float) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        with autocast_context(device, amp_dtype):
            output = model(motion)
            reconstruction = masked_smooth_l1(output["reconstruction"], motion, lengths)
            loss = reconstruction + commit_weight * output["commit"]
        total += float(loss) * len(motion)
        count += len(motion)
    tensor = torch.tensor([total, count], dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor)
    return float(tensor[0] / tensor[1].clamp_min(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    values = load_yaml_config(args.config)
    cfg = TokenizerConfig.from_mapping(values)
    world, rank, local, device = setup_distributed(int(values.get("seed", 20260628)))
    args.output.mkdir(parents=True, exist_ok=True)
    mean = np.load(args.prepared_root / "stats" / "mean133.npy")
    std = np.load(args.prepared_root / "stats" / "std133.npy")
    train_data = PreparedMotionDataset(args.prepared_root / "train.jsonl", mean, std, values.get("min_frames", 40), values.get("max_frames", 400))
    val_data = PreparedMotionDataset(args.prepared_root / "val.jsonl", mean, std, values.get("min_frames", 40), values.get("max_frames", 400))
    train_sampler = DistributedLengthBucketSampler(
        train_data.lengths, world, rank, True, int(values.get("seed", 20260628)),
        int(values.get("bucket_size", 2048)), False,
    )
    val_sampler = DistributedLengthBucketSampler(
        val_data.lengths, world, rank, False, int(values.get("seed", 20260628)),
        int(values.get("bucket_size", 2048)), False,
    )
    train_loader = DataLoader(train_data, batch_size=int(values.get("micro_batch", 240)), sampler=train_sampler, drop_last=True, collate_fn=collate_motion, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=int(values.get("eval_batch", 64)), sampler=val_sampler, collate_fn=collate_motion, num_workers=0)
    if device.type == "cuda" and bool(values.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = RobotSTARTokenizer(cfg).to(device)
    optimizer_kwargs = {
        "lr": float(values.get("learning_rate", 2e-4)),
        "betas": tuple(values.get("betas", [0.9, 0.99])),
        "weight_decay": float(values.get("weight_decay", 0.0)),
    }
    if device.type == "cuda" and bool(values.get("fused_adamw", True)):
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    amp_dtype = str(values.get("amp_dtype", "bf16"))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == "fp16")
    start_epoch = 0; step = 0; best = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)); step = int(checkpoint.get("step", 0)); best = float(checkpoint.get("best", best))
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False, gradient_as_bucket_view=True)
    epochs = int(values.get("epochs", 500))
    accumulation = int(values.get("gradient_accumulation", 1))
    total_updates = epochs * max(1, math.ceil(len(train_loader) / accumulation))
    commit_weight = float(values.get("commit_weight", 0.02))
    base_lr = float(values.get("learning_rate", 2e-4)); minimum_lr = 1e-6
    wandb_run = None
    if is_main() and args.wandb:
        import wandb
        wandb_run = wandb.init(project="RobotSTAR-Tokenizer", config={**values, "world_size": world})
    if is_main():
        write_json(args.output / "launch.json", {"config": values, "world_size": world, "train_samples": len(train_data), "val_samples": len(val_data)})
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train_loader, disable=not is_main(), desc=f"tokenizer {epoch + 1}/{epochs}")
        for iteration, batch in enumerate(bar):
            progress = min(1.0, step / max(1, total_updates - 1))
            learning_rate = minimum_lr + (base_lr - minimum_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            motion = batch["motion"].to(device)
            lengths = batch["lengths"].to(device)
            synchronize = (iteration + 1) % accumulation == 0 or (iteration + 1) == len(train_loader)
            context = model.no_sync() if isinstance(model, DDP) and not synchronize else torch.enable_grad()
            with context:
                with autocast_context(device, amp_dtype):
                    output = model(motion)
                    reconstruction = masked_smooth_l1(output["reconstruction"], motion, lengths)
                    loss = reconstruction + commit_weight * output["commit"]
                    scaled_loss = loss / accumulation
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            if synchronize:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(values.get("gradient_clip", 1.0)))
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True); step += 1
            if is_main():
                bar.set_postfix(loss=f"{float(loss):.4f}", rec=f"{float(reconstruction):.4f}")
                if wandb_run and synchronize:
                    wandb_run.log({"train/loss": float(loss), "train/reconstruction": float(reconstruction), "train/lr": learning_rate}, step=step)
        should_evaluate = (epoch + 1) % int(values.get("eval_every", 10)) == 0 or (epoch + 1) == epochs
        if should_evaluate:
            validation = evaluate(model, val_loader, device, amp_dtype, commit_weight)
            if is_main():
                payload = {
                    "version": "robotstar_fsq_tokenizer_v1",
                    "model": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": asdict(cfg),
                    "args": values,
                    "step": step,
                    "epoch": epoch + 1,
                    "best": min(best, validation),
                }
                save_checkpoint(args.output / "latest.pt", payload)
                if validation < best:
                    best = validation
                    payload["best"] = best
                    save_checkpoint(args.output / "best.pt", payload)
                if wandb_run:
                    wandb_run.log({"val/loss": validation}, step=step)
                print(f"[EPOCH {epoch + 1}] val={validation:.6f} best={best:.6f}")
        elif is_main() and (epoch + 1) % int(values.get("save_every", 10)) == 0:
            save_checkpoint(
                args.output / "latest.pt",
                {
                    "version": "robotstar_fsq_tokenizer_v1",
                    "model": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": asdict(cfg),
                    "args": values,
                    "step": step,
                    "epoch": epoch + 1,
                    "best": best,
                },
            )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
