from __future__ import annotations

from robotstar.retrieval import build_word_dictionary, extract_keywords


def main() -> None:
    rows = [
        {"word": "ABOUT", "source_id": "a", "reconstruction_mae": 0.5, "tokens": {"body": [1, 2], "left": [3, 4], "right": [5, 6]}},
        {"word": "ABOUT", "source_id": "b", "reconstruction_mae": 0.2, "tokens": {"body": [7, 8], "left": [9, 10], "right": [11, 12]}},
    ]
    dictionary = build_word_dictionary(rows)
    assert dictionary["ABOUT"]["source_id"] == "b"
    assert len(dictionary["ABOUT"]["body"]) == 10
    assert len(dictionary["ABOUT"]["lhand"]) == 10
    assert len(dictionary["ABOUT"]["rhand"]) == 10
    assert "left" not in dictionary["ABOUT"] and "right" not in dictionary["ABOUT"]
    assert extract_keywords("This is about signs.", dictionary) == ["ABOUT"]
    print("test_retrieval: PASS")


if __name__ == "__main__":
    main()
