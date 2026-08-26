from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import read_json, resolve_model_root
from .generator import RobotSTARGenerator
from .representation import denormalize_motion, resample_linear
from .retrieval import extract_keywords
from .tokenizer import RobotSTARTokenizer


def _length_tokens(mode: str, value: float, fps: float, maximum: int) -> int | None:
    if mode == "predicted":
        return None
    if value <= 0:
        raise ValueError("--length-value must be positive for explicit length modes")
    if mode == "tokens":
        tokens = int(round(value))
    elif mode == "frames":
        tokens = int(math.ceil(value / 4.0))
    elif mode == "seconds":
        tokens = int(math.ceil(value * fps / 4.0))
    else:
        raise ValueError(mode)
    return max(1, min(int(maximum), tokens))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Ivystream/RobotSTAR")
    parser.add_argument("--base-model", default="google/mt5-large")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length-mode", choices=["predicted", "tokens", "frames", "seconds"], default="predicted")
    parser.add_argument("--length-value", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-keywords", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = resolve_model_root(
        args.model,
        allow_patterns=["generator/*", "tokenizer/*", "stats/*", "retrieval/*", "README.md", "manifest.json"],
    )
    generator, vocab = RobotSTARGenerator.from_pretrained(root, base_model=args.base_model, device=device)
    generator.eval()
    tokenizer = RobotSTARTokenizer.from_pretrained(root, device=device).eval()
    mean = np.load(root / "stats" / "mean133.npy").astype(np.float32)
    std = np.load(root / "stats" / "std133.npy").astype(np.float32)

    dictionary: dict[str, Any] = {}
    dictionary_path = root / "retrieval" / "word2code.json"
    if args.retrieval and dictionary_path.is_file():
        dictionary = read_json(dictionary_path)
    keywords = extract_keywords(args.text, dictionary, args.max_keywords) if dictionary else []
    prompt = vocab.retrieval_prompt(args.text, keywords, dictionary) if dictionary else args.text
    encoded = vocab.encode_text([prompt], generator.cfg.max_source_length)
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    explicit_tokens = _length_tokens(args.length_mode, args.length_value, args.fps, generator.cfg.max_motion_tokens)
    explicit = None if explicit_tokens is None else torch.tensor([explicit_tokens], dtype=torch.long, device=device)
    with torch.inference_mode():
        generated = generator.generate(
            input_ids,
            attention,
            explicit,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=args.sample,
        )
        token_length = int(generated["full_lengths"][0].item())
        codes = {
            part: generated["final_codes"][part][0:1, :token_length]
            for part in ("body", "left", "right")
        }
        normalized = tokenizer.decode(codes)[0].float().cpu().numpy().astype(np.float32)
    desired_frames = None
    if args.length_mode == "frames":
        desired_frames = int(round(args.length_value))
    elif args.length_mode == "seconds":
        desired_frames = int(round(args.length_value * args.fps))
    if desired_frames and desired_frames != len(normalized):
        normalized = resample_linear(normalized, desired_frames)
    motion = denormalize_motion(normalized, mean, std)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "motion133_normalized.npy", normalized)
    np.save(args.output_dir / "motion133.npy", motion)
    np.savez_compressed(
        args.output_dir / "tokens.npz",
        body=codes["body"].cpu().numpy(),
        left=codes["left"].cpu().numpy(),
        right=codes["right"].cpu().numpy(),
        stage_lengths=generated["stage_lengths"].cpu().numpy(),
    )
    metadata = {
        "text": args.text,
        "prompt": prompt,
        "keywords": keywords,
        "retrieval_used": bool(dictionary and keywords),
        "length_mode": args.length_mode,
        "length_value": args.length_value,
        "motion_tokens": token_length,
        "frames": len(motion),
        "fps": args.fps,
        "seconds": len(motion) / args.fps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "sampling": args.sample,
        "seed": args.seed,
        "model": str(args.model),
        "base_model": str(args.base_model),
    }
    (args.output_dir / "generation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
