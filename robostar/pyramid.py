from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_STAGE_DIVISORS = (8, 4, 2, 1)


def parse_divisors(value: str | Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip()) if isinstance(value, str) else tuple(int(item) for item in value)
    if not values or values[-1] != 1 or any(item <= 0 for item in values):
        raise ValueError(f"Invalid stage divisors: {values}")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("Stage divisors must be strictly decreasing")
    return values


def stage_length(full_length: int, divisor: int) -> int:
    return max(1, math.ceil(int(full_length) / int(divisor)))


@torch.inference_mode()
def nearest_code(vectors: torch.Tensor, codebook: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    vectors = vectors.float()
    codebook = codebook.float()
    codebook_norm = codebook.square().sum(dim=-1).unsqueeze(0)
    outputs: list[torch.Tensor] = []
    for start in range(0, len(vectors), int(chunk_size)):
        block = vectors[start : start + int(chunk_size)]
        distance = block.square().sum(dim=-1, keepdim=True) - 2.0 * block @ codebook.t() + codebook_norm
        outputs.append(distance.argmin(dim=-1))
    return torch.cat(outputs) if outputs else torch.empty(0, dtype=torch.long, device=vectors.device)


@torch.inference_mode()
def coarse_codes(codes: torch.Tensor, codebook: torch.Tensor, output_length: int) -> torch.Tensor:
    codes = codes.long().flatten()
    if len(codes) == int(output_length):
        return codes.clone()
    embeddings = F.embedding(codes, codebook.float())
    pooled = F.adaptive_avg_pool1d(embeddings.t().unsqueeze(0), int(output_length)).squeeze(0).t()
    return nearest_code(pooled, codebook)


@torch.inference_mode()
def build_stream_pyramid(codes: torch.Tensor, codebook: torch.Tensor, divisors: Sequence[int]) -> list[torch.Tensor]:
    return [coarse_codes(codes, codebook, stage_length(len(codes), divisor)) for divisor in parse_divisors(divisors)]


@torch.inference_mode()
def build_triplet_pyramid(
    body: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    body_codebook: torch.Tensor,
    left_codebook: torch.Tensor,
    right_codebook: torch.Tensor,
    divisors: Sequence[int] = DEFAULT_STAGE_DIVISORS,
) -> list[dict[str, torch.Tensor]]:
    body_stages = build_stream_pyramid(body, body_codebook, divisors)
    left_stages = build_stream_pyramid(left, left_codebook, divisors)
    right_stages = build_stream_pyramid(right, right_codebook, divisors)
    return [
        {"body": body_stage, "left": left_stage, "right": right_stage}
        for body_stage, left_stage, right_stage in zip(body_stages, left_stages, right_stages)
    ]
