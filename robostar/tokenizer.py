from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .checkpoints import load_legacy_checkpoint, load_safetensors_directory, read_json, resolve_model_root
from .config import TokenizerConfig
from .fsq import ProductFSQ
from .representation import merge_motion133, split_motion133, validate_motion133
from .temporal import TemporalDecoder, TemporalEncoder


class MotionBranchTokenizer(nn.Module):
    def __init__(self, nfeats: int, code_num: int, levels: tuple[int, ...], cfg: TokenizerConfig) -> None:
        super().__init__()
        product = 1
        for level in levels:
            product *= int(level)
        if int(code_num) != product:
            raise ValueError("FSQ level product does not match code count")
        self.encoder = TemporalEncoder(
            nfeats,
            cfg.code_dim,
            cfg.down_t,
            cfg.stride_t,
            cfg.width,
            cfg.depth,
            cfg.dilation_growth_rate,
            activation="relu",
            norm=None,
        )
        self.quantizer = ProductFSQ(cfg.code_dim, levels)
        self.decoder = TemporalDecoder(
            nfeats,
            cfg.code_dim,
            cfg.down_t,
            cfg.stride_t,
            cfg.width,
            cfg.depth,
            cfg.dilation_growth_rate,
            activation="relu",
            norm=None,
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(features.permute(0, 2, 1))
        quantized, commitment, perplexity, _ = self.quantizer(latent)
        reconstruction = self.decoder(quantized).permute(0, 2, 1)
        return reconstruction, commitment, perplexity

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(features.permute(0, 2, 1))
        flat = latent.permute(0, 2, 1).reshape(-1, latent.shape[1])
        return self.quantizer.quantize(flat).view(features.shape[0], -1)

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        flat = self.quantizer.dequantize(ids.reshape(-1)).to(next(self.parameters()).dtype)
        latent = flat.view(ids.shape[0], ids.shape[1], -1).permute(0, 2, 1).contiguous()
        return self.decoder(latent).permute(0, 2, 1)

    @torch.no_grad()
    def codebook(self) -> torch.Tensor:
        return self.quantizer.all_codebook(next(self.parameters()).device)


class RoboSTARTokenizer(nn.Module):
    """Factorized body/left-hand/right-hand FSQ motion tokenizer."""

    def __init__(self, cfg: TokenizerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TokenizerConfig()
        self.body = MotionBranchTokenizer(43, self.cfg.body_code_num, self.cfg.body_levels, self.cfg)
        self.left = MotionBranchTokenizer(45, self.cfg.hand_code_num, self.cfg.hand_levels, self.cfg)
        self.right = MotionBranchTokenizer(45, self.cfg.hand_code_num, self.cfg.hand_levels, self.cfg)

    def forward(self, motion: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_motion133(motion)
        parts = split_motion133(motion)
        body_rec, body_commit, body_perplexity = self.body(parts["body"])
        left_rec, left_commit, left_perplexity = self.left(parts["left"])
        right_rec, right_commit, right_perplexity = self.right(parts["right"])
        return {
            "reconstruction": merge_motion133(body_rec, left_rec, right_rec),
            "commit": body_commit + left_commit + right_commit,
            "body_commit": body_commit,
            "left_commit": left_commit,
            "right_commit": right_commit,
            "body_perplexity": body_perplexity,
            "left_perplexity": left_perplexity,
            "right_perplexity": right_perplexity,
        }

    @torch.no_grad()
    def encode(self, motion: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_motion133(motion)
        parts = split_motion133(motion)
        return {
            "body": self.body.encode(parts["body"]),
            "left": self.left.encode(parts["left"]),
            "right": self.right.encode(parts["right"]),
        }

    @torch.no_grad()
    def decode(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return merge_motion133(
            self.body.decode(tokens["body"]),
            self.left.decode(tokens["left"]),
            self.right.decode(tokens["right"]),
        )

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "RoboSTARTokenizer":
        path = Path(model)
        if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}:
            checkpoint = load_legacy_checkpoint(path, device="cpu")
            instance = cls(TokenizerConfig.from_mapping(checkpoint.get("config", {})))
            missing, unexpected = instance.load_state_dict(checkpoint["model"], strict=strict)
        else:
            root = resolve_model_root(model, allow_patterns=["tokenizer/*", "manifest.json", "README.md"])
            folder = root / "tokenizer" if (root / "tokenizer").is_dir() else root
            instance = cls(TokenizerConfig.from_mapping(read_json(folder / "config.json")))
            state = load_safetensors_directory(folder, device="cpu")
            missing, unexpected = instance.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Tokenizer state mismatch: missing={missing}, unexpected={unexpected}")
        return instance.to(device)
