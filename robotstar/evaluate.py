from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .data import read_jsonl


def dtw_path(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = cdist(left.astype(np.float64), right.astype(np.float64), metric="euclidean")
    table = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            table[i, j] = cost[i - 1, j - 1] + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    i, j = len(left), len(right)
    path_left: list[int] = []
    path_right: list[int] = []
    while i > 0 and j > 0:
        path_left.append(i - 1)
        path_right.append(j - 1)
        move = int(np.argmin((table[i - 1, j - 1], table[i - 1, j], table[i, j - 1])))
        if move == 0:
            i -= 1; j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return np.asarray(path_left[::-1]), np.asarray(path_right[::-1])


def sequence_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pi, ti = dtw_path(prediction, target)
    aligned_pred = prediction[pi]
    aligned_target = target[ti]
    error = np.abs(aligned_pred - aligned_target)
    pred_velocity = np.diff(prediction, axis=0)
    target_velocity = np.diff(target, axis=0)
    velocity_pi, velocity_ti = dtw_path(pred_velocity, target_velocity)
    speed = np.linalg.norm(pred_velocity, axis=-1)
    acceleration = np.diff(pred_velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    tail = speed[max(0, int(0.8 * len(speed))):]
    return {
        "dtw_mae_133": float(error.mean()),
        "dtw_body43_mae": float(np.abs(np.concatenate((aligned_pred[:, :30], aligned_pred[:, 120:133]), axis=-1) - np.concatenate((aligned_target[:, :30], aligned_target[:, 120:133]), axis=-1)).mean()),
        "dtw_left45_mae": float(np.abs(aligned_pred[:, 30:75] - aligned_target[:, 30:75]).mean()),
        "dtw_right45_mae": float(np.abs(aligned_pred[:, 75:120] - aligned_target[:, 75:120]).mean()),
        "velocity_mae": float(np.abs(pred_velocity[velocity_pi] - target_velocity[velocity_ti]).mean()),
        "length_abs_error_frames": float(abs(len(prediction) - len(target))),
        "length_ratio": float(len(prediction) / max(len(target), 1)),
        "freeze_tail_rate": float((tail < 1e-3).mean()) if len(tail) else 0.0,
        "near_static_rate": float((speed < 1e-3).mean()) if len(speed) else 0.0,
        "jitter_rms": float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="Directory containing <source_id>.npy or <source_id>/motion133.npy")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(args.manifest):
        source_id = str(item["source_id"])
        candidates = [args.predictions / f"{source_id}.npy", args.predictions / source_id / "motion133.npy"]
        prediction_path = next((path for path in candidates if path.is_file()), None)
        if prediction_path is None:
            continue
        prediction = np.load(prediction_path).astype(np.float32)
        target = np.load(item["motion133_npy"]).astype(np.float32)
        rows.append({"source_id": source_id, **sequence_metrics(prediction, target)})
    if not rows:
        raise RuntimeError("No matching predictions")
    keys = [key for key in rows[0] if key != "source_id"]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    payload = {"samples": len(rows), "summary": summary, "per_sample": rows}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
