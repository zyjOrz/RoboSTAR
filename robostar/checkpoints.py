from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def resolve_model_root(model: str | Path, allow_patterns: Iterable[str] | None = None) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()
    downloaded = snapshot_download(repo_id=str(model), allow_patterns=list(allow_patterns) if allow_patterns else None)
    return Path(downloaded).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_safetensors_directory(directory: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    folder = Path(directory)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        mapping = read_json(index).get("weight_map", {})
        files = sorted({str(name) for name in mapping.values()})
    else:
        direct = folder / "model.safetensors"
        if direct.is_file():
            files = [direct.name]
        else:
            files = sorted(path.name for path in folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files under {folder}")
    state: dict[str, torch.Tensor] = {}
    for filename in files:
        shard = load_file(str(folder / filename), device=str(device))
        overlap = set(state).intersection(shard)
        if overlap:
            raise RuntimeError(f"Duplicate state keys in shards: {sorted(overlap)[:5]}")
        state.update(shard)
    return state


def load_legacy_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError(f"Unsupported checkpoint schema: {path}")
    return checkpoint
