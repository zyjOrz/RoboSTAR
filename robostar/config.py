from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


def _filter_dataclass(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


@dataclass
class TokenizerConfig:
    body_code_num: int = 96
    hand_code_num: int = 192
    body_levels: tuple[int, ...] = (4, 4, 6)
    hand_levels: tuple[int, ...] = (4, 6, 8)
    code_dim: int = 512
    width: int = 512
    depth: int = 3
    down_t: int = 2
    stride_t: int = 2
    dilation_growth_rate: int = 3

    def __post_init__(self) -> None:
        self.body_levels = tuple(int(x) for x in self.body_levels)
        self.hand_levels = tuple(int(x) for x in self.hand_levels)
        if _product(self.body_levels) != int(self.body_code_num):
            raise ValueError("body_levels do not match body_code_num")
        if _product(self.hand_levels) != int(self.hand_code_num):
            raise ValueError("hand_levels do not match hand_code_num")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TokenizerConfig":
        return cls(**_filter_dataclass(cls, values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratorConfig:
    model_path: str = "google/mt5-large"
    body_codes: int = 96
    hand_codes: int = 192
    alpha_hand: float = 0.4
    max_source_length: int = 512
    max_motion_tokens: int = 100
    use_compact_map: bool = True
    attn_implementation: str = "eager"
    stage_divisors: tuple[int, ...] = (8, 4, 2, 1)
    label_smoothing: float = 0.0
    stage_loss_weights: tuple[float, ...] = (1.25, 1.10, 1.00, 1.25)
    hand_loss_weight: float = 1.25
    length_loss_weight: float = 0.20
    contrastive_loss_weight: float = 0.03
    contrastive_temperature: float = 0.07
    lm_family: str = "mt5"
    scale_decoder_outputs: bool = True

    def __post_init__(self) -> None:
        self.stage_divisors = tuple(int(x) for x in self.stage_divisors)
        self.stage_loss_weights = tuple(float(x) for x in self.stage_loss_weights)
        if not self.stage_divisors or self.stage_divisors[-1] != 1:
            raise ValueError("stage_divisors must end in 1")
        if any(a <= b for a, b in zip(self.stage_divisors, self.stage_divisors[1:])):
            raise ValueError("stage_divisors must be strictly decreasing")
        if len(self.stage_loss_weights) != len(self.stage_divisors):
            raise ValueError("stage_loss_weights must match stage_divisors")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GeneratorConfig":
        return cls(**_filter_dataclass(cls, values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result
