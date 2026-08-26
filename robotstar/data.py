from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .representation import canonicalize_length, normalize_motion
from .vocabulary import RobotSTARVocabulary


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            result.append(row)
    return result


class DistributedLengthBucketSampler(Sampler[int]):
    """Deterministic distributed sampler that reduces variable-length padding."""

    def __init__(
        self,
        lengths: Sequence[int],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        bucket_size: int = 1024,
        drop_last: bool = False,
    ) -> None:
        self.lengths = [int(value) for value in lengths]
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(1, int(bucket_size))
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank={rank} is invalid for num_replicas={num_replicas}")
        if self.drop_last:
            self.num_samples = len(self.lengths) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
        for bucket in buckets:
            bucket.sort(key=self.lengths.__getitem__)
            if self.shuffle and rng.random() < 0.5:
                bucket.reverse()
        if self.shuffle:
            rng.shuffle(buckets)
        ordered = [index for bucket in buckets for index in bucket]
        if self.drop_last:
            ordered = ordered[: self.total_size]
        elif ordered:
            padding = self.total_size - len(ordered)
            if padding > 0:
                repeats = math.ceil(padding / len(ordered))
                ordered.extend((ordered * repeats)[:padding])
        return iter(ordered[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


class PreparedMotionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        mean: np.ndarray,
        std: np.ndarray,
        minimum_frames: int = 40,
        maximum_frames: int = 400,
    ) -> None:
        self.rows = read_jsonl(manifest)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.minimum_frames = int(minimum_frames)
        self.maximum_frames = int(maximum_frames)
        self.lengths = [int(row.get("num_frames", 0)) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        motion = np.load(row["motion133_npy"]).astype(np.float32)
        motion = canonicalize_length(motion, self.minimum_frames, self.maximum_frames, 4)
        normalized = normalize_motion(motion, self.mean, self.std)
        return {
            "source_id": str(row.get("source_id", index)),
            "text": str(row.get("text", "")),
            "motion": torch.from_numpy(normalized),
            "length": len(normalized),
            "row": row,
        }


def collate_motion(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    length = max(int(item["length"]) for item in batch)
    motion = torch.zeros((len(batch), length, 133), dtype=torch.float32)
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    for index, item in enumerate(batch):
        motion[index, : item["length"]] = item["motion"]
    return {
        "motion": motion,
        "lengths": lengths,
        "source_ids": [item["source_id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


class GeneratorCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
        vocab: RobotSTARVocabulary,
        train: bool,
        retrieval_dropout: float = 0.10,
        edge_drop_augmentation: bool = True,
    ) -> None:
        cache = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        if cache.get("version") != "robotstar_c2f_cache_v1":
            raise RuntimeError(f"Unsupported cache version: {cache.get('version')}")
        self.rows = cache["rows"]
        self.vocab = vocab
        self.train = bool(train)
        self.retrieval_dropout = float(retrieval_dropout)
        self.edge_drop_augmentation = bool(edge_drop_augmentation)
        self.stage_divisors = tuple(int(value) for value in cache["stage_divisors"])
        self.lengths = [int(row["variants"][0]["full_length"]) for row in self.rows]
        self._materialize()

    def _materialize(self) -> None:
        for row in self.rows:
            for key in ("input_ids_plain", "input_ids_retrieval"):
                if not torch.is_tensor(row[key]):
                    row[key] = torch.as_tensor(row[key], dtype=torch.long)
            for variant in row["variants"]:
                if not torch.is_tensor(variant["stage_lengths"]):
                    variant["stage_lengths"] = torch.as_tensor(variant["stage_lengths"], dtype=torch.long)
                for stage in variant["stages"]:
                    for part in ("body", "left", "right"):
                        if not torch.is_tensor(stage[part]):
                            stage[part] = torch.as_tensor(stage[part], dtype=torch.long)
                if "stage_labels" not in variant:
                    variant["stage_labels"] = [
                        {
                            part: torch.tensor(self.vocab.labels(stage[part].tolist(), part), dtype=torch.long)
                            for part in ("body", "left", "right")
                        }
                        for stage in variant["stages"]
                    ]

    def __len__(self) -> int:
        return len(self.rows)

    def _choose_variant(self, row: dict[str, Any]) -> dict[str, Any]:
        variants = row["variants"]
        if not self.train or not self.edge_drop_augmentation or len(variants) == 1:
            return variants[0]
        draw = random.randrange(6)
        if draw < 4:
            return variants[0]
        return variants[1] if draw == 4 else variants[min(2, len(variants) - 1)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        variant = self._choose_variant(row)
        use_plain = self.train and random.random() < self.retrieval_dropout
        return {
            "source_id": str(row["source_id"]),
            "text": str(row.get("text", "")),
            "input_ids": row["input_ids_plain"] if use_plain else row["input_ids_retrieval"],
            "full_length": int(variant["full_length"]),
            "stage_lengths": variant["stage_lengths"].long(),
            "stages": variant["stages"],
            "stage_labels": variant["stage_labels"],
            "used_retrieval": not use_plain,
        }


class GeneratorCollator:
    def __init__(self, vocab: RobotSTARVocabulary) -> None:
        self.vocab = vocab

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        source_length = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((batch_size, source_length), self.vocab.pad_id, dtype=torch.long)
        attention = torch.zeros((batch_size, source_length), dtype=torch.bool)
        for index, item in enumerate(batch):
            size = len(item["input_ids"])
            input_ids[index, :size] = item["input_ids"]
            attention[index, :size] = True
        stage_codes: list[dict[str, torch.Tensor]] = []
        stage_labels: list[dict[str, torch.Tensor]] = []
        stage_count = len(batch[0]["stages"])
        for stage_index in range(stage_count):
            maximum = max(int(item["stage_lengths"][stage_index]) for item in batch)
            codes = {part: torch.full((batch_size, maximum), -100, dtype=torch.long) for part in ("body", "left", "right")}
            labels = {part: torch.full((batch_size, maximum + 2), -100, dtype=torch.long) for part in ("body", "left", "right")}
            for row_index, item in enumerate(batch):
                for part in ("body", "left", "right"):
                    sequence = item["stages"][stage_index][part]
                    label = item["stage_labels"][stage_index][part]
                    codes[part][row_index, : len(sequence)] = sequence
                    labels[part][row_index, : len(label)] = label
            stage_codes.append(codes)
            stage_labels.append(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "full_lengths": torch.tensor([item["full_length"] for item in batch], dtype=torch.long),
            "length_classes": torch.tensor([item["full_length"] - 1 for item in batch], dtype=torch.long),
            "stage_lengths": torch.stack([item["stage_lengths"] for item in batch]),
            "stage_codes": stage_codes,
            "stage_labels": stage_labels,
            "source_ids": [item["source_id"] for item in batch],
            "texts": [item["text"] for item in batch],
        }
