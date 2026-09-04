from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import read_jsonl
from .pyramid import build_triplet_pyramid, parse_divisors
from .retrieval import extract_keywords
from .tokenizer import RoboSTARTokenizer
from .vocabulary import RoboSTARVocabulary, token_contract


def _resample_token_triplet(row: dict[str, Any], minimum: int = 10, maximum: int = 100) -> torch.Tensor:
    streams = row.get("tokens", row)
    body = list(streams["body"])
    left = list(streams["left"])
    right = list(streams["right"])
    length = min(len(body), len(left), len(right))
    target = min(max(length, minimum), maximum)
    indices = [min(length - 1, int(index * length / target)) for index in range(target)]
    return torch.tensor([[body[i], left[i], right[i]] for i in indices], dtype=torch.long)


def _variants(row: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    base = _resample_token_triplet(row)
    result = [("base", base)]
    if len(base) > 2:
        result.extend((("drop_head", base[1:]), ("drop_tail", base[:-1])))
    return result


def _trim(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ids[: int(mask.long().sum().item())].to(torch.int32).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--base-model", default="google/mt5-large")
    parser.add_argument("--retrieval", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--stage-divisors", default="8,4,2,1")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-keywords", type=int, default=3)
    parser.add_argument("--max-len-per-part", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = RoboSTARTokenizer.from_pretrained(args.tokenizer_model, device=device).eval()
    vocab = RoboSTARVocabulary(args.base_model, 96, 192, True)
    dictionary = json.loads(args.retrieval.read_text(encoding="utf-8")) if args.retrieval else {}
    divisors = parse_divisors(args.stage_divisors)
    codebooks = {
        "body": tokenizer.body.codebook().detach(),
        "left": tokenizer.left.codebook().detach(),
        "right": tokenizer.right.codebook().detach(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        source = args.token_root / f"{split}_source_tokens.jsonl"
        rows_in = read_jsonl(source)
        texts = [str(row.get("text", "")) for row in rows_in]
        prompts = []
        for text in texts:
            keywords = extract_keywords(text, dictionary, args.max_keywords) if dictionary else []
            prompts.append(vocab.retrieval_prompt(text, keywords, dictionary, args.max_len_per_part) if dictionary else text)
        plain_ids: list[torch.Tensor] = []
        retrieval_ids: list[torch.Tensor] = []
        for start in tqdm(range(0, len(rows_in), 256), desc=f"tokenize {split}"):
            plain = vocab.encode_text(texts[start : start + 256], args.max_source_length)
            retrieved = vocab.encode_text(prompts[start : start + 256], args.max_source_length)
            plain_ids.extend(_trim(ids, mask) for ids, mask in zip(plain["input_ids"], plain["attention_mask"]))
            retrieval_ids.extend(_trim(ids, mask) for ids, mask in zip(retrieved["input_ids"], retrieved["attention_mask"]))
        rows_out: list[dict[str, Any]] = []
        for index, row in enumerate(tqdm(rows_in, desc=f"pyramid {split}")):
            variants = []
            for name, triplet in _variants(row):
                pyramid = build_triplet_pyramid(
                    triplet[:, 0].to(device),
                    triplet[:, 1].to(device),
                    triplet[:, 2].to(device),
                    codebooks["body"],
                    codebooks["left"],
                    codebooks["right"],
                    divisors,
                )
                variants.append(
                    {
                        "name": name,
                        "full_length": len(triplet),
                        "stage_lengths": [len(stage["body"]) for stage in pyramid],
                        "stages": [
                            {part: stage[part].cpu().to(torch.int16) for part in ("body", "left", "right")}
                            for stage in pyramid
                        ],
                    }
                )
            rows_out.append(
                {
                    "source_id": str(row["source_id"]),
                    "text": str(row.get("text", "")),
                    "prompt_retrieval": prompts[index],
                    "input_ids_plain": plain_ids[index],
                    "input_ids_retrieval": retrieval_ids[index],
                    "variants": variants,
                    "num_frames": int(row.get("num_frames", 0)),
                }
            )
        target = args.output / f"{split}.pt"
        torch.save(
            {
                "version": "robostar_c2f_cache_v1",
                "rows": rows_out,
                "split": split,
                "stage_divisors": list(divisors),
                "vocab": vocab.config(),
                "token_contract": token_contract(),
            },
            target,
        )
        print(f"[SAVE] {target} rows={len(rows_out)}")


if __name__ == "__main__":
    main()
