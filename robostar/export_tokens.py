from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .data import read_jsonl
from .representation import normalize_motion
from .tokenizer import RoboSTARTokenizer


def _distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return world, rank, device


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    world, rank, device = _distributed()
    tokenizer = RoboSTARTokenizer.from_pretrained(args.model, device=device).eval()
    mean = np.load(args.prepared_root / "stats" / "mean133.npy").astype(np.float32)
    std = np.load(args.prepared_root / "stats" / "std133.npy").astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        rows = read_jsonl(args.prepared_root / f"{split}.jsonl")
        local_rows = [(index, row) for index, row in enumerate(rows) if index % world == rank]
        exported: list[dict[str, Any]] = []
        for start in range(0, len(local_rows), args.batch_size):
            block = local_rows[start : start + args.batch_size]
            arrays = [normalize_motion(np.load(row["motion133_npy"]).astype(np.float32), mean, std) for _, row in block]
            lengths = [len(array) for array in arrays]
            maximum = max(lengths)
            batch = torch.zeros((len(block), maximum, 133), dtype=torch.float32, device=device)
            for batch_index, array in enumerate(arrays):
                batch[batch_index, : len(array)] = torch.from_numpy(array).to(device)
            with torch.inference_mode():
                encoded = tokenizer.encode(batch)
                reconstructed = tokenizer.decode(encoded)
            for batch_index, (original_index, row) in enumerate(block):
                token_length = lengths[batch_index] // 4
                reconstruction = reconstructed[batch_index, : lengths[batch_index]]
                target = batch[batch_index, : lengths[batch_index]]
                exported.append(
                    {
                        "source_id": str(row["source_id"]),
                        "text": str(row.get("text", "")),
                        "num_frames": lengths[batch_index],
                        "fps": float(row.get("fps", 20.0)),
                        "motion133_npy": str(row["motion133_npy"]),
                        "reconstruction_mae": float((reconstruction - target).abs().mean().item()),
                        "tokens": {
                            "body": encoded["body"][batch_index, :token_length].cpu().tolist(),
                            "left": encoded["left"][batch_index, :token_length].cpu().tolist(),
                            "right": encoded["right"][batch_index, :token_length].cpu().tolist(),
                        },
                        "_index": original_index,
                    }
                )
        shard = args.output / f"{split}.rank{rank:04d}.jsonl"
        _write_jsonl(shard, exported)
        if world > 1:
            dist.barrier()
        if rank == 0:
            merged: list[dict[str, Any]] = []
            for shard_rank in range(world):
                merged.extend(read_jsonl(args.output / f"{split}.rank{shard_rank:04d}.jsonl"))
            merged.sort(key=lambda row: int(row.pop("_index")))
            final = args.output / f"{split}_source_tokens.jsonl"
            _write_jsonl(final, merged)
            print(f"[SAVE] {final} rows={len(merged)}")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
