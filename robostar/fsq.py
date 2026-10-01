from __future__ import annotations

from math import prod
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ProductFSQ(nn.Module):
    """Finite scalar quantization with mixed-radix integer codes."""

    def __init__(self, code_dim: int, levels: Sequence[int]) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.levels = tuple(int(value) for value in levels)
        if not self.levels or any(value < 2 for value in self.levels):
            raise ValueError(f"Invalid FSQ levels: {levels}")
        self.num_digits = len(self.levels)
        self.code_num = int(prod(self.levels))
        self.project_in = nn.Linear(self.code_dim, self.num_digits)
        self.project_out = nn.Linear(self.num_digits, self.code_dim)
        radix: list[int] = []
        value = 1
        for level in self.levels:
            radix.append(value)
            value *= level
        self.register_buffer("radix", torch.tensor(radix, dtype=torch.long), persistent=False)
        self.register_buffer("levels_t", torch.tensor(self.levels, dtype=torch.float32), persistent=False)
        self.register_buffer("levels_l", torch.tensor(self.levels, dtype=torch.long), persistent=False)

    def _digits_to_ids(self, digits: torch.Tensor) -> torch.Tensor:
        return (digits.long() * self.radix.to(digits.device)).sum(dim=-1)

    def _ids_to_digits(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.long()
        columns = [
            torch.div(ids, radix, rounding_mode="floor") % level
            for radix, level in zip(self.radix.to(ids.device), self.levels_l.to(ids.device))
        ]
        return torch.stack(columns, dim=-1)

    def _digits_to_normalized(self, digits: torch.Tensor) -> torch.Tensor:
        levels = self.levels_t.to(device=digits.device, dtype=torch.float32)
        return digits.float() / (levels - 1.0) * 2.0 - 1.0

    def quantize(self, vectors: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.project_in(vectors.float()))
        levels = self.levels_t.to(projected.device)
        grid = (projected + 1.0) * 0.5 * (levels - 1.0)
        digits = torch.minimum(grid.round().clamp_min(0), levels - 1.0).long()
        return self._digits_to_ids(digits)

    def dequantize(self, ids: torch.Tensor) -> torch.Tensor:
        normalized = self._digits_to_normalized(self._ids_to_digits(ids))
        return self.project_out(normalized.to(self.project_out.weight.dtype))

    def all_codebook(self, device: torch.device | None = None) -> torch.Tensor:
        target = device if device is not None else self.project_out.weight.device
        return self.dequantize(torch.arange(self.code_num, device=target, dtype=torch.long)).float()

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, time = value.shape
        flat = value.permute(0, 2, 1).reshape(-1, channels).float()
        projected = torch.tanh(self.project_in(flat))
        levels = self.levels_t.to(projected.device)
        grid = (projected + 1.0) * 0.5 * (levels - 1.0)
        rounded = torch.minimum(grid.round().clamp_min(0), levels - 1.0)
        straight_through = grid + (rounded - grid).detach()
        ids = self._digits_to_ids(rounded.long())
        normalized = straight_through / (levels - 1.0) * 2.0 - 1.0
        quantized = self.project_out(normalized.to(self.project_out.weight.dtype)).float()
        commitment = F.mse_loss(flat, quantized.detach()) + F.mse_loss(quantized, flat.detach())
        quantized = flat + (quantized - flat).detach() + (quantized - quantized.detach())
        counts = torch.bincount(ids.detach(), minlength=self.code_num).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probabilities * torch.log(probabilities + 1e-7)).sum())
        quantized = quantized.view(batch, time, channels).permute(0, 2, 1).contiguous().to(value.dtype)
        return quantized, commitment, perplexity, ids.view(batch, time)
