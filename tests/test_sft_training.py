from __future__ import annotations

import json
from pathlib import Path

from src.training.sft import (
    SFTDataCollator,
    SFTDataset,
    encode_sft_row,
    load_sft_rows,
    weighted_causal_lm_loss,
)


class FakeTokenizer:
    eos_token_id = 9

    @staticmethod
    def encode(value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 7 + 1 for character in value]

    def __call__(
        self, value: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": self.encode(value, add_special_tokens=False),
            "offset_mapping": [(index, index + 1) for index in range(len(value))],
        }


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


def test_weighted_encoding_marks_reasoning_code_boundaries_and_eos() -> None:
    reasoning = "reason"
    code = "int main(){}"
    response = f"<think>\n{reasoning}\n</think>\n\n```cpp\n{code}\n```"
    row = {
        "problem_id": "weighted",
        "prompt": "prompt",
        "response": response,
        "reasoning": reasoning,
        "code": code,
    }
    weights = {"reasoning": 0.25, "code": 1.0, "boundary": 2.0, "eos": 4.0}
    encoded = encode_sft_row(row, FakeTokenizer(), max_length=100, loss_weights=weights)
    prompt_length = len(row["prompt"])
    response_weights = encoded["loss_weights"][prompt_length:-1]
    reasoning_start = response.index(reasoning)
    code_start = response.index(code)
    assert encoded["loss_weights"][:prompt_length] == [0.0] * prompt_length
    assert response_weights[reasoning_start] == 0.25
    assert response_weights[code_start] == 1.0
    assert response_weights[0] == 2.0
    assert encoded["loss_weights"][-1] == 4.0


def test_weighted_causal_loss_uses_target_token_weights() -> None:
    import torch

    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    weights = torch.tensor([[0.0, 1.0, 3.0]])
    loss = weighted_causal_lm_loss(logits, labels, weights)
    expected = (
        torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 1], reduction="none")
        + 3 * torch.nn.functional.cross_entropy(logits[:, 1], labels[:, 2], reduction="none")
    ) / 4
    assert torch.allclose(loss, expected.mean())
