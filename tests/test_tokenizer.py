from __future__ import annotations

import torch

from robotstar.config import TokenizerConfig
from robotstar.tokenizer import RobotSTARTokenizer


def main() -> None:
    torch.manual_seed(0)
    cfg = TokenizerConfig(
        body_code_num=8,
        hand_code_num=8,
        body_levels=(2, 2, 2),
        hand_levels=(2, 2, 2),
        code_dim=16,
        width=16,
        depth=1,
        down_t=2,
        stride_t=2,
        dilation_growth_rate=3,
    )
    model = RobotSTARTokenizer(cfg).eval()
    x = torch.randn(2, 40, 133)
    tokens = model.encode(x)
    assert tokens["body"].shape == (2, 10)
    assert tokens["left"].shape == (2, 10)
    assert tokens["right"].shape == (2, 10)
    reconstruction = model.decode(tokens)
    assert reconstruction.shape == x.shape
    state = model.state_dict()
    expected = {
        "body.encoder.model.0.weight",
        "body.quantizer.project_in.weight",
        "body.quantizer.project_out.weight",
        "body.decoder.model.6.weight",
        "left.encoder.model.0.weight",
        "right.decoder.model.6.weight",
    }
    missing = expected.difference(state)
    assert not missing, missing
    print("test_tokenizer: PASS")


if __name__ == "__main__":
    main()
