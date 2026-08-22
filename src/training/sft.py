from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def load_sft_rows(path: str | Path, *, limit: int | None, selection: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if selection == "ordered":
        selected = rows
    elif selection == "shortest":
        selected = sorted(rows, key=lambda row: (int(row["token_counts"]["total"]), row["problem_id"]))
    else:
        raise ValueError(f"Unsupported data selection: {selection}")
    return selected if limit is None else selected[:limit]


def encode_sft_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    input_ids = prompt_ids + response_ids + [int(eos_token_id)]
    if len(input_ids) > max_length:
        raise ValueError(
            f"{row['problem_id']} has {len(input_ids)} tokens including EOS; max_length={max_length}"
        )
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + response_ids + [int(eos_token_id)],
        "attention_mask": [1] * len(input_ids),
        "problem_id": row["problem_id"],
        "length": len(input_ids),
    }


class SFTDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_length: int,
        limit: int | None = None,
        selection: str = "ordered",
    ) -> None:
        rows = load_sft_rows(path, limit=limit, selection=selection)
        if not rows:
            raise ValueError("SFT dataset is empty")
        self.examples = [encode_sft_row(row, tokenizer, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class SFTDataCollator:
    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of > 1:
            maximum = (
                (maximum + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
