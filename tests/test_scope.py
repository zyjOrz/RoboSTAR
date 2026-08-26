from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_files = ("mujoco", "wuji", "unitree", "retarget", "transcribe", "speech_to_text")
    for path in (root / "robotstar").rglob("*.py"):
        lowered = path.name.lower()
        assert not any(token in lowered for token in forbidden_files), path
        text = path.read_text(encoding="utf-8").lower()
        for pattern in ("import mujoco", "import unitree", "import wuji", "from wuji", "from unitree"):
            assert pattern not in text, (path, pattern)
    print("test_scope: PASS")


if __name__ == "__main__":
    main()
