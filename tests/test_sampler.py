from __future__ import annotations

from robotstar.data import DistributedLengthBucketSampler


def main() -> None:
    lengths = [9, 2, 8, 1, 7, 3, 6, 4, 5]
    rank0 = list(DistributedLengthBucketSampler(lengths, 2, 0, True, 7, 4, False))
    rank1 = list(DistributedLengthBucketSampler(lengths, 2, 1, True, 7, 4, False))
    assert len(rank0) == len(rank1) == 5
    assert all(0 <= index < len(lengths) for index in rank0 + rank1)
    repeat = list(DistributedLengthBucketSampler(lengths, 2, 0, True, 7, 4, False))
    assert rank0 == repeat
    print("test_sampler: PASS")


if __name__ == "__main__":
    main()
