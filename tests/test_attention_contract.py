from robotstar.config import GeneratorConfig
from robotstar.generator import resolve_attention_implementation


def main() -> None:
    cfg = GeneratorConfig()
    assert cfg.attn_implementation == "eager"
    assert resolve_attention_implementation("sdpa", "mt5") == "eager"
    assert resolve_attention_implementation("eager", "mt5") == "eager"
    assert resolve_attention_implementation("sdpa", "t5") == "sdpa"
    print("test_attention_contract: PASS")


if __name__ == "__main__":
    main()
