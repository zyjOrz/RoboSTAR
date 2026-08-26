from __future__ import annotations

import numpy as np
import torch

from robotstar.representation import merge_motion133, split_motion133


def main() -> None:
    rng = np.random.default_rng(7)
    array = rng.normal(size=(11, 133)).astype(np.float32)
    parts = split_motion133(array)
    np.testing.assert_array_equal(merge_motion133(parts["body"], parts["left"], parts["right"]), array)
    tensor = torch.from_numpy(array)
    parts_t = split_motion133(tensor)
    torch.testing.assert_close(merge_motion133(parts_t["body"], parts_t["left"], parts_t["right"]), tensor)
    print("test_representation: PASS")


if __name__ == "__main__":
    main()
