from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import read_jsonl
from .representation import canonicalize_length, validate_motion133


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_split(source_manifest: Path, output_root: Path, split: str) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows_out: list[dict[str, Any]] = []
    motions: list[np.ndarray] = []
    sequence_root = output_root / split / "seqs"
    sequence_root.mkdir(parents=True, exist_ok=True)
    for row in read_jsonl(source_manifest):
        source_id = str(row["source_id"])
        source_path = Path(row["motion133_npy"])
        motion = np.load(source_path).astype(np.float32)
        validate_motion133(motion)
        motion = canonicalize_length(motion, 40, 400, 4)
        destination = sequence_root / f"{source_id}.npy"
        np.save(destination, motion)
        rows_out.append(
            {
                "source_id": source_id,
                "text": str(row.get("text", "")),
                "motion133_npy": str(destination.resolve()),
                "num_frames": len(motion),
                "fps": float(row.get("fps", 20.0)),
            }
        )
        motions.append(motion)
    write_jsonl(output_root / f"{split}.jsonl", rows_out)
    return rows_out, motions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, training = prepare_split(args.train_manifest, args.output, "train")
    prepare_split(args.val_manifest, args.output, "val")
    prepare_split(args.test_manifest, args.output, "test")
    if not training:
        raise RuntimeError("Training manifest is empty")
    stacked = np.concatenate(training, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    stats = args.output / "stats"
    stats.mkdir(exist_ok=True)
    np.save(stats / "mean133.npy", mean)
    np.save(stats / "std133.npy", std)
    print(json.dumps({"output": str(args.output), "train_frames": len(stacked)}, indent=2))


if __name__ == "__main__":
    main()
