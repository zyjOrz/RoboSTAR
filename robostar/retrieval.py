from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_keyword(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def simple_lemma(value: str) -> str:
    word = normalize_keyword(value)
    for suffix in ("IES", "ING", "ED", "ES", "S"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            if suffix == "IES":
                return word[:-3] + "Y"
            return word[: -len(suffix)]
    return word


def extract_keywords(text: str, dictionary: dict[str, Any], maximum: int = 3) -> list[str]:
    result: list[str] = []
    for token in re.findall(r"[A-Za-z]+", str(text)):
        for candidate in (normalize_keyword(token), simple_lemma(token)):
            if candidate in dictionary and candidate not in result:
                result.append(candidate)
                break
        if len(result) >= int(maximum):
            break
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _fixed_length(values: Iterable[int], size: int = 10) -> list[int]:
    sequence = [int(value) for value in values]
    if not sequence:
        return [0] * int(size)
    if len(sequence) == int(size):
        return sequence
    result = []
    for index in range(int(size)):
        source = min(len(sequence) - 1, int(index * len(sequence) / int(size)))
        result.append(sequence[source])
    return result


def build_word_dictionary(rows: Iterable[dict[str, Any]], prototype_length: int = 10) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        word = normalize_keyword(str(row.get("word", row.get("label", row.get("text", "")))))
        tokens = row.get("tokens", row)
        if not word or not isinstance(tokens, dict):
            continue
        left_key = "left" if "left" in tokens else "lhand" if "lhand" in tokens else None
        right_key = "right" if "right" in tokens else "rhand" if "rhand" in tokens else None
        if "body" not in tokens or left_key is None or right_key is None:
            continue
        normalized = dict(row)
        normalized["tokens"] = {
            "body": tokens["body"],
            "lhand": tokens[left_key],
            "rhand": tokens[right_key],
        }
        grouped[word].append(normalized)
    output: dict[str, Any] = {}
    for word, candidates in sorted(grouped.items()):
        best = min(candidates, key=lambda row: float(row.get("reconstruction_mae", float("inf"))))
        tokens = best.get("tokens", best)
        output[word] = {
            "word": word,
            "source_id": str(best.get("source_id", "")),
            "reconstruction_mae": float(best.get("reconstruction_mae", float("nan"))),
            "num_candidates": len(candidates),
            "body": _fixed_length(tokens["body"], prototype_length),
            "lhand": _fixed_length(tokens["lhand"], prototype_length),
            "rhand": _fixed_length(tokens["rhand"], prototype_length),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a word-to-FSQ-code memory.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--word-token-jsonl", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--prototype-length", type=int, default=10)
    args = parser.parse_args()
    if args.command == "build":
        dictionary = build_word_dictionary(_read_jsonl(args.word_token_jsonl), args.prototype_length)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({"output": str(args.output), "entries": len(dictionary)}, indent=2))


if __name__ == "__main__":
    main()
