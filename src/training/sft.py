from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _find_span(text: str, value: str, *, field: str) -> tuple[int, int]:
    value = value.strip()
    start = text.find(value)
    if not value or start < 0:
        raise ValueError(f"Response does not contain its {field} field")
    return start, start + len(value)


def _weighted_response_encoding(
    row: dict[str, Any], tokenizer: Any, weights: dict[str, Any]
) -> tuple[list[int], list[float]]:
    response = row["response"]
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(response_ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent response offsets")
    reasoning_span = _find_span(response, row.get("reasoning", ""), field="reasoning")
    code_span = _find_span(response, row.get("code", ""), field="code")
    reasoning_weight = float(weights["reasoning"])
    code_weight = float(weights["code"])
    boundary_weight = float(weights["boundary"])
    if min(reasoning_weight, code_weight, boundary_weight) < 0:
        raise ValueError("Loss weights must be non-negative")

    token_weights = []
    for start, end in offsets:
        if start < reasoning_span[1] and end > reasoning_span[0]:
            token_weights.append(reasoning_weight)
        elif start < code_span[1] and end > code_span[0]:
            token_weights.append(code_weight)
        else:
            token_weights.append(boundary_weight)
    return response_ids, token_weights

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


def encode_sft_row(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    loss_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if row.get("messages") is not None:
        if loss_weights is not None:
            raise ValueError("Agent message SFT does not support reasoning/code loss weights")
        return encode_agent_sft_row(row, tokenizer, max_length)
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    if loss_weights is None:
        response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        response_weights = None
    else:
        response_ids, response_weights = _weighted_response_encoding(row, tokenizer, loss_weights)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    input_ids = prompt_ids + response_ids + [int(eos_token_id)]
    if len(input_ids) > max_length:
        raise ValueError(
            f"{row['problem_id']} has {len(input_ids)} tokens including EOS; max_length={max_length}"
        )
    encoded = {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + response_ids + [int(eos_token_id)],
        "attention_mask": [1] * len(input_ids),
        "problem_id": row["problem_id"],
        "length": len(input_ids),
    }
    if response_weights is not None:
        eos_weight = float(loss_weights["eos"])
        if eos_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        encoded["loss_weights"] = [0.0] * len(prompt_ids) + response_weights + [eos_weight]
    return encoded


def encode_agent_sft_row(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Agent SFT row requires non-empty messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Agent SFT conversation must end with an assistant message")

    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported Agent SFT role: {role}")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise ValueError("Agent SFT messages require non-empty string content")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    assistant_spans = []
    cursor = 0
    for message in messages:
        start = rendered.find(message["content"], cursor)
        if start < 0:
            raise ValueError("Chat template output does not contain message content in order")
        end = start + len(message["content"])
        if message["role"] == "assistant":
            # Qwen chat templates terminate every assistant turn with
            # <|im_end|>. Supervise that terminator as well as the textual
            # action so the fine-tuned policy learns to stop each tool call
            # and final answer instead of continuing indefinitely.
            terminator = rendered.find("<|im_end|>", end)
            if terminator >= 0:
                end = terminator + len("<|im_end|>")
            assistant_spans.append((start, end))
        cursor = end
    labels = [
        token
        if any(start < span_end and end > span_start for span_start, span_end in assistant_spans)
        else -100
        for token, (start, end) in zip(input_ids, offsets)
    ]
    if len(input_ids) > max_length:
        raise ValueError(
            f"{row['problem_id']} has {len(input_ids)} chat tokens; max_length={max_length}"
        )
    if not any(label != -100 for label in labels):
        raise ValueError("Agent SFT conversation has no assistant target tokens")
    return {
        "input_ids": input_ids,
        "labels": labels,
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
        loss_weights: dict[str, Any] | None = None,
    ) -> None:
        rows = load_sft_rows(path, limit=limit, selection=selection)
        if not rows:
            raise ValueError("SFT dataset is empty")
        self.examples = [
            encode_sft_row(row, tokenizer, max_length, loss_weights=loss_weights) for row in rows
        ]

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
        weighted = "loss_weights" in features[0]
        if any(("loss_weights" in feature) != weighted for feature in features):
            raise ValueError("Cannot mix weighted and unweighted SFT examples")
        if weighted:
            batch["loss_weights"] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            if weighted:
                batch["loss_weights"].append(feature["loss_weights"] + [0.0] * padding)
        return {
            key: torch.tensor(value, dtype=torch.float32 if key == "loss_weights" else torch.long)
            for key, value in batch.items()
        }


def weighted_causal_lm_loss(logits: Any, labels: Any, loss_weights: Any) -> Any:
    """Compute globally normalized weighted next-token cross entropy under DDP."""

    import torch
    import torch.nn.functional as functional

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].contiguous().float()
    active = shift_labels.ne(-100)
    effective_weights = shift_weights * active
    per_token = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    local_numerator = (per_token * effective_weights).sum()
    local_denominator = effective_weights.sum()
    if local_denominator.item() <= 0:
        raise ValueError("Weighted SFT batch has no positive-weight target tokens")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_denominator = local_denominator.detach().clone()
        global_numerator = local_numerator.detach().clone()
        torch.distributed.all_reduce(global_denominator)
        torch.distributed.all_reduce(global_numerator)
        world_size = torch.distributed.get_world_size()
        gradient_loss = local_numerator * world_size / global_denominator
        reported_loss = global_numerator / global_denominator
        return gradient_loss + (reported_loss - gradient_loss.detach())
    return local_numerator / local_denominator
