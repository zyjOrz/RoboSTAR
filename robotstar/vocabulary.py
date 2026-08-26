from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


@dataclass
class VocabularyConfig:
    model_path: str = "google/mt5-large"
    body_codes: int = 96
    hand_codes: int = 192
    use_compact_map: bool = True


class RobotSTARVocabulary:
    """mT5 tokenizer extended with atomic motion-code symbols.

    Motion symbols use the RobotSTAR namespace. They are added to the base
    mT5 tokenizer in a fixed order; release validation verifies that the
    resulting token IDs match the pretrained checkpoint embedding rows.
    """

    def __init__(
        self,
        model_path: str,
        body_codes: int = 96,
        hand_codes: int = 192,
        use_compact_map: bool = True,
    ) -> None:
        from transformers import AutoTokenizer

        self.cfg = VocabularyConfig(str(model_path), int(body_codes), int(hand_codes), bool(use_compact_map))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        language_token = "<robotstar:en_asl>"
        body = [f"<robotstar:body:{index}>" for index in range(self.cfg.body_codes)]
        left = [f"<robotstar:left:{index}>" for index in range(self.cfg.hand_codes)]
        right = [f"<robotstar:right:{index}>" for index in range(self.cfg.hand_codes)]
        self.en_asl_token = language_token
        self.body_tokens = body
        self.left_tokens = left
        self.right_tokens = right
        requested = [language_token, *body, *left, *right]
        existing = set(self.tokenizer.get_vocab())
        additions = [token for token in requested if token not in existing]
        if additions:
            self.tokenizer.add_tokens(additions, special_tokens=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        if self.tokenizer.eos_token_id is None:
            raise RuntimeError("mT5 tokenizer has no EOS token")
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.en_asl_id = int(self.tokenizer.convert_tokens_to_ids(language_token))
        self.body_code_to_id = [int(self.tokenizer.convert_tokens_to_ids(token)) for token in body]
        self.left_code_to_id = [int(self.tokenizer.convert_tokens_to_ids(token)) for token in left]
        self.right_code_to_id = [int(self.tokenizer.convert_tokens_to_ids(token)) for token in right]
        self.body_id_to_code = {token_id: code for code, token_id in enumerate(self.body_code_to_id)}
        self.left_id_to_code = {token_id: code for code, token_id in enumerate(self.left_code_to_id)}
        self.right_id_to_code = {token_id: code for code, token_id in enumerate(self.right_code_to_id)}
        self.vocab_size = int(len(self.tokenizer))

    def config(self) -> dict[str, Any]:
        return {
            **asdict(self.cfg),
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "en_asl_id": self.en_asl_id,
            "token_contract": token_contract(),
        }

    def code_token_ids(self, part: str) -> list[int]:
        if part == "body":
            return self.body_code_to_id
        if part == "left":
            return self.left_code_to_id
        if part == "right":
            return self.right_code_to_id
        raise KeyError(part)

    def labels(self, codes: Sequence[int], part: str) -> list[int]:
        table = self.code_token_ids(part)
        result = [table[int(code)] for code in codes if int(code) >= 0]
        result.extend((self.eos_id, self.en_asl_id))
        return result

    def encode_text(self, texts: Sequence[str], max_source_length: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [str(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=int(max_source_length),
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"].long(), "attention_mask": encoded["attention_mask"].long()}

    def retrieval_prompt(
        self,
        text: str,
        keywords: Sequence[str],
        word2code: dict[str, Any],
        max_len_per_part: int = 10,
    ) -> str:
        lines = [f"English: {str(text).strip()}"]
        hints: list[str] = []
        for keyword in keywords:
            entry = word2code.get(keyword)
            if entry is None:
                continue
            streams = entry.get("tokens", entry) if isinstance(entry, dict) else entry
            if isinstance(streams, dict):
                values = []
                aliases = {
                    "body": ("body",),
                    "left": ("left", "lhand"),
                    "right": ("right", "rhand"),
                }
                for part, names in aliases.items():
                    sequence = []
                    for name in names:
                        if name in streams:
                            sequence = list(streams[name])[: int(max_len_per_part)]
                            break
                    values.append(f"{part}=" + " ".join(str(value) for value in sequence))
                hints.append(f"- {keyword}: " + "; ".join(values))
        if hints:
            lines.append("Retrieved sign-token hints:")
            lines.extend(hints)
        return "\n".join(lines)


def token_contract() -> dict[str, Any]:
    return {
        "version": "robotstar_fsq_c2f_mt5_single_code_v1",
        "frame_dim": 133,
        "body_dim": 43,
        "left_dim": 45,
        "right_dim": 45,
        "body_vocab": 96,
        "hand_vocab": 192,
        "stage_divisors": [8, 4, 2, 1],
        "temporal_downsample": 4,
        "token_namespace": "robotstar",
    }
