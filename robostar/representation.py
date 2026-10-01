from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class MotionLayout:
    frame_dim: int = 133
    upper_body: slice = slice(0, 30)
    left_hand: slice = slice(30, 75)
    right_hand: slice = slice(75, 120)
    jaw: slice = slice(120, 123)
    expression: slice = slice(123, 133)


LAYOUT = MotionLayout()


def validate_motion133(motion: np.ndarray | torch.Tensor) -> None:
    if motion.ndim < 2 or int(motion.shape[-1]) != LAYOUT.frame_dim:
        raise ValueError(f"Expected [..., T, 133] or [T, 133], got {tuple(motion.shape)}")


def split_motion133(motion: np.ndarray | torch.Tensor) -> Mapping[str, np.ndarray | torch.Tensor]:
    validate_motion133(motion)
    if torch.is_tensor(motion):
        body = torch.cat((motion[..., :30], motion[..., 120:133]), dim=-1)
    else:
        body = np.concatenate((motion[..., :30], motion[..., 120:133]), axis=-1)
    return {
        "body": body,
        "left": motion[..., 30:75],
        "right": motion[..., 75:120],
    }


def merge_motion133(
    body: np.ndarray | torch.Tensor,
    left: np.ndarray | torch.Tensor,
    right: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    if body.shape[-1] != 43 or left.shape[-1] != 45 or right.shape[-1] != 45:
        raise ValueError("Expected body43, left45, and right45")
    if torch.is_tensor(body):
        return torch.cat((body[..., :30], left, right, body[..., 30:43]), dim=-1)
    return np.concatenate((body[..., :30], left, right, body[..., 30:43]), axis=-1)


def normalize_motion(motion: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    validate_motion133(motion)
    if mean.shape != (133,) or std.shape != (133,):
        raise ValueError("mean and std must be 133D")
    return ((motion.astype(np.float32) - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-10)).astype(np.float32)


def denormalize_motion(motion: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    validate_motion133(motion)
    return (motion.astype(np.float32) * (std.astype(np.float32) + 1e-10) + mean.astype(np.float32)).astype(np.float32)


def resample_linear(motion: np.ndarray, target_frames: int) -> np.ndarray:
    validate_motion133(motion)
    target_frames = int(target_frames)
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    if len(motion) == target_frames:
        return np.asarray(motion, dtype=np.float32).copy()
    old = np.linspace(0.0, 1.0, len(motion), dtype=np.float64)
    new = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    result = np.empty((target_frames, motion.shape[1]), dtype=np.float32)
    for column in range(motion.shape[1]):
        result[:, column] = np.interp(new, old, motion[:, column]).astype(np.float32)
    return result


def canonical_frame_length(length: int, minimum: int = 40, maximum: int = 400, multiple: int = 4) -> int:
    value = min(max(int(length), int(minimum)), int(maximum))
    return max(int(multiple), int(math.floor(value / multiple) * multiple))


def canonicalize_length(
    motion: np.ndarray,
    minimum: int = 40,
    maximum: int = 400,
    multiple: int = 4,
) -> np.ndarray:
    target = canonical_frame_length(len(motion), minimum, maximum, multiple)
    return resample_linear(np.asarray(motion, dtype=np.float32), target)


def token_length_from_frames(frames: int, temporal_downsample: int = 4) -> int:
    return max(1, int(frames) // int(temporal_downsample))
