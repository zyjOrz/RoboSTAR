from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import GeneratorConfig
from .data import DistributedLengthBucketSampler, GeneratorCacheDataset, GeneratorCollator
from .generator import RoboSTARGenerator
from .training import (
    autocast_context,
    cosine_with_warmup,
    is_main,
    load_yaml_config,
    move_to_device,
    ramp_probability,
    save_checkpoint,
    setup_distributed,
    unwrap,
    write_json,
)
from .vocabulary import RoboSTARVocabulary, token_contract


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: str) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        with autocast_context(device, amp_dtype):
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["full_lengths"],
                batch["length_classes"],
                batch["stage_lengths"],
                batch["stage_codes"],
                batch["stage_labels"],
                0.0,
                0.0,
            )
        size = len(batch["input_ids"])
        count += size
        for key, value in output.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[key] = totals.get(key, 0.0) + float(value) * size
    keys = sorted(totals)
    tensor = torch.tensor([*[totals[key] for key in keys], count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    denominator = float(tensor[-1].clamp_min(1))
    return {key: float(tensor[index] / denominator) for index, key in enumerate(keys)}


def _transition_rate(values: torch.Tensor, length: int) -> float:
    if int(length) <= 1:
        return 0.0
    segment = values[: int(length)]
    return float(segment[1:].ne(segment[:-1]).float().mean())


def _stream_accuracy(target: torch.Tensor, prediction: torch.Tensor, length: int) -> float:
    if int(length) <= 0:
        return 0.0
    return float(target[: int(length)].eq(prediction[: int(length)]).float().mean())


@torch.inference_mode()
def free_evaluate(
    model: RoboSTARGenerator,
    dataset: GeneratorCacheDataset,
    collator: GeneratorCollator,
    device: torch.device,
    maximum_sources: int,
    batch_size: int,
    rank: int,
    world: int,
) -> dict[str, float]:
    """Greedy free-generation validation used to select ``best_generation.pt``."""
    model.eval()
    count = min(len(dataset), int(maximum_sources)) if int(maximum_sources) > 0 else len(dataset)
    indices = list(range(count))[int(rank) :: max(1, int(world))]
    loader = DataLoader(Subset(dataset, indices), batch_size=int(batch_size), shuffle=False, collate_fn=collator)
    totals: dict[str, float] = {}
    local_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        full_lengths = batch["full_lengths"].to(device, non_blocking=True)
        predicted = model.generate(input_ids, attention, None, 1.0, 0, False)
        oracle = model.generate(input_ids, attention, full_lengths, 1.0, 0, False)
        for index in range(len(input_ids)):
            gt_length = int(full_lengths[index])
            predicted_length = int(predicted["full_lengths"][index])
            totals["length_mae"] = totals.get("length_mae", 0.0) + abs(predicted_length - gt_length)
            for prefix, output in (("pred", predicted), ("oracle", oracle)):
                compare_length = min(gt_length, int(output["full_lengths"][index]))
                accuracies: list[float] = []
                transition_errors: list[float] = []
                for part in ("body", "left", "right"):
                    target = batch["stage_codes"][-1][part][index]
                    generated = output["final_codes"][part][index].cpu()
                    accuracies.append(_stream_accuracy(target, generated, compare_length))
                    transition_errors.append(
                        abs(
                            _transition_rate(target, gt_length)
                            - _transition_rate(generated, int(output["full_lengths"][index]))
                        )
                    )
                totals[f"{prefix}_final_acc"] = totals.get(f"{prefix}_final_acc", 0.0) + float(np.mean(accuracies))
                totals[f"{prefix}_transition_error"] = totals.get(f"{prefix}_transition_error", 0.0) + float(np.mean(transition_errors))
            for stage_index in range(model.nstages):
                stage_length = int(batch["stage_lengths"][index, stage_index])
                accuracies = []
                for part in ("body", "left", "right"):
                    target = batch["stage_codes"][stage_index][part][index]
                    generated = oracle["stages"][stage_index][part][index].cpu()
                    accuracies.append(_stream_accuracy(target, generated, stage_length))
                key = f"oracle_stage_{stage_index + 1}_acc"
                totals[key] = totals.get(key, 0.0) + float(np.mean(accuracies))
            local_count += 1
    if dist.is_initialized():
        gathered: list[dict[str, Any] | None] = [None for _ in range(world)]
        dist.all_gather_object(gathered, {"totals": totals, "count": local_count})
        merged: dict[str, float] = {}
        total_count = 0
        for item in gathered:
            if not item:
                continue
            total_count += int(item["count"])
            for key, value in item["totals"].items():
                merged[key] = merged.get(key, 0.0) + float(value)
        totals, local_count = merged, total_count
    metrics = {key: value / max(1, local_count) for key, value in totals.items()}
    metrics["num_sources"] = float(local_count)
    metrics["selection_score"] = (
        0.55 * metrics.get("pred_final_acc", 0.0)
        + 0.35 * metrics.get("oracle_final_acc", 0.0)
        + 0.10 * metrics.get("oracle_stage_1_acc", 0.0)
        - 0.002 * metrics.get("length_mae", 0.0)
        - 0.10 * metrics.get("pred_transition_error", 0.0)
    )
    return metrics


def _sync_stop(stop: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return bool(stop)
    flag = torch.tensor([1 if stop else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: GeneratorConfig,
    values: dict[str, Any],
    vocab: RoboSTARVocabulary,
    step: int,
    epoch: int,
    best: float,
    best_generation: float,
) -> dict[str, Any]:
    return {
        "version": "robostar_fsq_c2f_mt5_large_v1",
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": asdict(cfg),
        "args": values,
        "vocab": vocab.config(),
        "token_contract": token_contract(),
        "step": int(step),
        "epoch": int(epoch),
        "best": float(best),
        "best_generation": float(best_generation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    values = load_yaml_config(args.config)
    cfg = GeneratorConfig.from_mapping(values)
    world, rank, local, device = setup_distributed(int(values.get("seed", 20260625)))
    if device.type == "cuda" and bool(values.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)

    vocab = RoboSTARVocabulary(
        cfg.model_path,
        cfg.body_codes,
        cfg.hand_codes,
        cfg.use_compact_map,
    )
    train_data = GeneratorCacheDataset(
        args.cache_root / "train.pt", vocab, True, float(values.get("retrieval_dropout", 0.10)), True
    )
    val_data = GeneratorCacheDataset(args.cache_root / "val.pt", vocab, False, 0.0, False)
    seed = int(values.get("seed", 20260625))
    bucket_size = int(values.get("bucket_size", 512))
    train_sampler = DistributedLengthBucketSampler(train_data.lengths, world, rank, True, seed, bucket_size, False)
    val_sampler = DistributedLengthBucketSampler(val_data.lengths, world, rank, False, seed, bucket_size, False)
    collator = GeneratorCollator(vocab)
    train_loader = DataLoader(
        train_data,
        batch_size=int(values.get("micro_batch", 8)),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(values.get("eval_batch", 8)),
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=0,
    )

    model = RoboSTARGenerator(cfg, vocab, backbone_init="pretrained").to(device)
    backbone: list[torch.nn.Parameter] = []
    new_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("main_lm.") else new_parameters).append(parameter)
    optimizer_kwargs: dict[str, Any] = {
        "betas": tuple(values.get("betas", [0.9, 0.99])),
        "weight_decay": float(values.get("weight_decay", 0.01)),
    }
    if device.type == "cuda" and bool(values.get("fused_adamw", True)):
        optimizer_kwargs["fused"] = True
    parameter_groups = [
        {"params": backbone, "lr": float(values.get("backbone_lr", 2e-5))},
        {"params": new_parameters, "lr": float(values.get("new_lr", 1e-4))},
    ]
    try:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)

    amp_dtype = str(values.get("amp_dtype", "bf16"))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == "fp16")
    start_epoch = 0
    step = 0
    best = float("inf")
    best_generation = float("-inf")
    no_improve = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        step = int(checkpoint.get("step", 0))
        best = float(checkpoint.get("best", best))
        best_generation = float(checkpoint.get("best_generation", best_generation))

    if world > 1:
        model = DDP(
            model,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            bucket_cap_mb=int(values.get("ddp_bucket_cap_mb", 64)),
        )

    accumulation = int(values.get("gradient_accumulation", 3))
    epochs = int(values.get("epochs", 500))
    max_steps = int(values.get("max_steps", 100000))
    total_updates = min(max_steps, epochs * max(1, math.ceil(len(train_loader) / accumulation))) if max_steps > 0 else epochs * max(1, math.ceil(len(train_loader) / accumulation))
    warmup = int(values.get("warmup_steps", 1000))
    minimum_ratio = float(values.get("eta_min_ratio", 0.05))
    backbone_lr = float(values.get("backbone_lr", 2e-5))
    new_lr = float(values.get("new_lr", 1e-4))

    wandb_run = None
    if is_main() and args.wandb:
        import wandb

        wandb_run = wandb.init(project="RoboSTAR", config={**values, "world_size": world})
    if is_main():
        write_json(
            args.output / "launch.json",
            {
                "config": values,
                "model": asdict(cfg),
                "vocab": vocab.config(),
                "world_size": world,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
            },
        )

    optimizer.zero_grad(set_to_none=True)
    stop = False
    last_epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch + 1
        train_sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train_loader, disable=not is_main(), desc=f"generator {epoch + 1}/{epochs}")
        for iteration, batch in enumerate(bar):
            ratio = cosine_with_warmup(step, total_updates, warmup, minimum_ratio)
            optimizer.param_groups[0]["lr"] = backbone_lr * ratio
            optimizer.param_groups[1]["lr"] = new_lr * ratio
            self_condition = ramp_probability(
                step,
                int(values.get("self_condition_warmup", 2000)),
                int(values.get("self_condition_ramp", 6000)),
                float(values.get("self_condition_max", 0.15)),
            )
            corruption = ramp_probability(
                step,
                int(values.get("context_corruption_warmup", 1000)),
                int(values.get("context_corruption_ramp", 5000)),
                float(values.get("context_corruption_max", 0.12)),
            )
            batch = move_to_device(batch, device)
            synchronize = (iteration + 1) % accumulation == 0 or (iteration + 1) == len(train_loader)
            context = model.no_sync() if isinstance(model, DDP) and not synchronize else torch.enable_grad()
            with context:
                with autocast_context(device, amp_dtype):
                    output = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["full_lengths"],
                        batch["length_classes"],
                        batch["stage_lengths"],
                        batch["stage_codes"],
                        batch["stage_labels"],
                        corruption,
                        self_condition,
                    )
                    loss = output["loss"] / accumulation
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if synchronize:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(values.get("gradient_clip", 1.0)))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if is_main() and wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": float(output["loss"]),
                            "train/token_loss": float(output["token_loss"]),
                            "train/length_mae": float(output["length_mae"]),
                            "train/self_condition": self_condition,
                            "train/context_corruption": corruption,
                            "train/grad_norm": float(gradient),
                            "train/backbone_lr": optimizer.param_groups[0]["lr"],
                            "train/new_lr": optimizer.param_groups[1]["lr"],
                        },
                        step=step,
                    )
            if is_main():
                bar.set_postfix(loss=f"{float(output['loss']):.4f}", length=f"{float(output['length_mae']):.2f}")
            if max_steps > 0 and step >= max_steps:
                stop = True
                break

        should_evaluate = (epoch + 1) % int(values.get("eval_every", 1)) == 0 or stop or (epoch + 1) == epochs
        if should_evaluate:
            validation = evaluate(model, val_loader, device, amp_dtype)
            if is_main():
                payload = _checkpoint_payload(model, optimizer, scaler, cfg, values, vocab, step, epoch + 1, best, best_generation)
                save_checkpoint(args.output / "latest.pt", payload)
                if validation["loss"] < best:
                    best = validation["loss"]
                    payload = _checkpoint_payload(model, optimizer, scaler, cfg, values, vocab, step, epoch + 1, best, best_generation)
                    save_checkpoint(args.output / "best.pt", payload)
                print(f"[EPOCH {epoch + 1}] {validation} best={best:.6f}")
                if wandb_run:
                    wandb_run.log({f"val/{key}": value for key, value in validation.items()}, step=step)

        should_free_evaluate = int(values.get("free_eval_every", 6)) > 0 and (
            (epoch + 1) % int(values.get("free_eval_every", 6)) == 0 or stop or (epoch + 1) == epochs
        )
        if should_free_evaluate:
            if dist.is_initialized():
                dist.barrier()
            free_metrics = free_evaluate(
                unwrap(model),
                val_data,
                collator,
                device,
                int(values.get("free_eval_sources", 48)),
                int(values.get("free_eval_batch", 4)),
                rank,
                world,
            )
            if is_main():
                print(f"[FREE EPOCH {epoch + 1}] {free_metrics}")
                if wandb_run:
                    wandb_run.log({f"free_val/{key}": value for key, value in free_metrics.items()}, step=step)
                if free_metrics["selection_score"] > best_generation:
                    best_generation = free_metrics["selection_score"]
                    no_improve = 0
                    save_checkpoint(
                        args.output / "best_generation.pt",
                        _checkpoint_payload(model, optimizer, scaler, cfg, values, vocab, step, epoch + 1, best, best_generation),
                    )
                else:
                    no_improve += 1
                    if no_improve >= int(values.get("early_stop_patience", 10)):
                        stop = True
            stop = _sync_stop(stop, device)
            if dist.is_initialized():
                dist.barrier()
        elif is_main() and (epoch + 1) % int(values.get("save_every", 2)) == 0:
            save_checkpoint(
                args.output / "latest.pt",
                _checkpoint_payload(model, optimizer, scaler, cfg, values, vocab, step, epoch + 1, best, best_generation),
            )

        stop = _sync_stop(stop, device)
        if stop:
            break

    if is_main():
        save_checkpoint(
            args.output / "latest.pt",
            _checkpoint_payload(model, optimizer, scaler, cfg, values, vocab, step, last_epoch, best, best_generation),
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
