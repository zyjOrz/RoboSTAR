from __future__ import annotations

import torch

from robotstar.pyramid import build_stream_pyramid


def main() -> None:
    torch.manual_seed(0)
    codes = torch.arange(80) % 8
    codebook = torch.randn(8, 16)
    stages = build_stream_pyramid(codes, codebook, (8, 4, 2, 1))
    assert [len(stage) for stage in stages] == [10, 20, 40, 80]
    torch.testing.assert_close(stages[-1], codes)
    print("test_pyramid: PASS")


if __name__ == "__main__":
    main()
