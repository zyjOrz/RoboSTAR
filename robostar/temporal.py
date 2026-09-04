from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    key = name.lower()
    if key == "relu":
        return nn.ReLU
    if key == "gelu":
        return nn.GELU
    if key in {"silu", "swish"}:
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


def _normalization(name: str | None, channels: int) -> nn.Module:
    if name is None:
        return nn.Identity()
    key = name.upper()
    if key == "LN":
        return nn.LayerNorm(channels)
    if key == "GN":
        return nn.GroupNorm(32, channels, eps=1e-6)
    if key == "BN":
        return nn.BatchNorm1d(channels, eps=1e-6)
    raise ValueError(f"Unsupported normalization: {name}")


class TemporalResidualUnit(nn.Module):
    """A pre-activation dilated temporal residual unit.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        dilation: int,
        activation: str = "relu",
        norm: str | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = _normalization(norm, channels)
        self.norm2 = _normalization(norm, channels)
        self.norm = norm
        act = _activation(activation)
        self.act1 = act()
        self.act2 = act()
        self.conv1 = nn.Conv1d(channels, hidden_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def _apply_norm(self, value: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        if self.norm == "LN":
            return layer(value.transpose(-2, -1)).transpose(-2, -1)
        return layer(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self._apply_norm(value, self.norm1)
        value = self.conv1(self.act1(value))
        value = self._apply_norm(value, self.norm2)
        value = self.conv2(self.act2(value))
        return value + residual


class TemporalResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        dilation_growth_rate: int = 3,
        reverse_dilation: bool = True,
        activation: str = "relu",
        norm: str | None = None,
    ) -> None:
        super().__init__()
        dilations = [int(dilation_growth_rate) ** index for index in range(int(depth))]
        if reverse_dilation:
            dilations.reverse()
        self.model = nn.Sequential(
            *[
                TemporalResidualUnit(channels, channels, dilation, activation=activation, norm=norm)
                for dilation in dilations
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm: str | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(nfeats, width, 3, 1, 1), nn.ReLU()]
        kernel = int(stride_t) * 2
        padding = int(stride_t) // 2
        for _ in range(int(down_t)):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(width, width, kernel, stride_t, padding),
                    TemporalResidualStack(
                        width,
                        depth,
                        dilation_growth_rate,
                        reverse_dilation=True,
                        activation=activation,
                        norm=norm,
                    ),
                )
            )
        layers.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm: str | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(output_emb_width, width, 3, 1, 1), nn.ReLU()]
        for _ in range(int(down_t)):
            layers.append(
                nn.Sequential(
                    TemporalResidualStack(
                        width,
                        depth,
                        dilation_growth_rate,
                        reverse_dilation=True,
                        activation=activation,
                        norm=norm,
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(width, width, 3, 1, 1),
                )
            )
        layers.extend([nn.Conv1d(width, width, 3, 1, 1), nn.ReLU(), nn.Conv1d(width, nfeats, 3, 1, 1)])
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)
