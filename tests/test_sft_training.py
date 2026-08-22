from __future__ import annotations

import json
from pathlib import Path

from src.training.sft import SFTDataCollator, SFTDataset, encode_sft_row, load_sft_rows


class FakeTokenizer:
    eos_token_id = 9

    @staticmethod
    def encode(value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 7 + 1 for character in value]


def _row(problem_id: str, total: int = 10) -> dict:
    return {
        "problem_id": problem_id,
        "prompt": "prompt",
        "response": "response",
        "token_counts": {"total": total},
    }


def test_encoding_masks_prompt_and_keeps_response_and_eos() -> None:
    encoded = encode_sft_row(_row("a"), FakeTokenizer(), max_length=100)
    prompt_length = len(FakeTokenizer.encode("prompt", add_special_tokens=False))
    assert encoded["labels"][:prompt_length] == [-100] * prompt_length
    assert encoded["labels"][prompt_length:] == encoded["input_ids"][prompt_length:]
    assert encoded["input_ids"][-1] == FakeTokenizer.eos_token_id


def test_encoding_rejects_instead_of_truncating() -> None:
    try:
        encode_sft_row(_row("too-long"), FakeTokenizer(), max_length=2)
    except ValueError as error:
        assert "too-long" in str(error)
    else:
        raise AssertionError("Expected an over-length sample error")


def test_dataset_selection_and_collation(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("torch")
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [_row("long", 20), _row("short", 5)]))
    assert [row["problem_id"] for row in load_sft_rows(path, limit=1, selection="shortest")] == ["short"]
    dataset = SFTDataset(path, FakeTokenizer(), max_length=100, limit=2, selection="shortest")
    batch = SFTDataCollator(0, pad_to_multiple_of=8)([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 16)
    assert (batch["labels"][:, :6] == -100).all()
