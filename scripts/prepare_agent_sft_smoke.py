from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.data.agent_eval import read_jsonl, write_jsonl
from src.training.sft import encode_agent_sft_row


def canonical_response(submission: dict[str, Any], *, action: str | None = None) -> str:
    action = action or submission["effective_action"]
    response = submission["response"].strip()
    prefix = f"<action>{action}</action>"
    return response if response.startswith(prefix) else f"{prefix}\n{response}"


INITIAL_SUBMISSION_MARKER = "\n\nAn earlier student submitted the following GNU C++17 program:"


def problem_only(user_prompt: str) -> str:
    if INITIAL_SUBMISSION_MARKER not in user_prompt:
        raise ValueError("Repair prompt does not contain the initial-submission marker")
    problem, remainder = user_prompt.split(INITIAL_SUBMISSION_MARKER, 1)
    if not problem.strip() or "It received this real execution-environment observation:" not in remainder:
        raise ValueError("Repair prompt cannot be separated into problem and failure context")
    return problem.strip()


def feedback_with_budget(feedback: str, remaining: int) -> str:
    updated, count = re.subn(
        r"(?m)^- Executions remaining: \d+$",
        f"- Executions remaining: {remaining}",
        feedback,
    )
    if count == 0:
        updated = f"{feedback.rstrip()}\n- Executions remaining: {remaining}"
    elif count != 1:
        raise ValueError("Execution feedback contains multiple remaining-budget lines")
    return updated


def trajectory_messages(row: dict[str, Any], *, max_execute_calls: int = 3) -> list[dict[str, Any]]:
    steps = row["repair_trajectory"]["steps"]
    if not steps:
        raise ValueError(f"{row['task_id']}: trajectory has no steps")
    final = steps[-1]
    if final["submission"]["effective_action"] != "final":
        raise ValueError(f"{row['task_id']}: trajectory does not end in final")
    teacher_execute_calls = sum(
        step["submission"]["effective_action"] == "execute_code" for step in steps
    )
    total_execute_calls = 1 + teacher_execute_calls
    if total_execute_calls > max_execute_calls:
        raise ValueError(
            f"{row['task_id']}: {total_execute_calls} execute calls exceed Agent budget "
            f"{max_execute_calls}"
        )

    initial_prompt = steps[0]["prompt_messages"]
    system_messages = [message for message in initial_prompt if message["role"] == "system"]
    user_messages = [message for message in initial_prompt if message["role"] == "user"]
    if len(system_messages) != 1 or len(user_messages) != 1 or len(initial_prompt) != 2:
        raise ValueError(f"{row['task_id']}: unexpected initial prompt structure")
    initial_code = row["initial_submission"]["code"].strip()
    initial_feedback = row["initial_observation"]["model_feedback"]
    messages = [
        dict(system_messages[0]),
        {"role": "user", "content": problem_only(user_messages[0]["content"])},
        {
            "role": "assistant",
            "content": f"<action>execute_code</action>\n```cpp\n{initial_code}\n```",
            "trainable": False,
        },
        {
            "role": "tool",
            "content": feedback_with_budget(initial_feedback, max_execute_calls - 1),
        },
    ]
    execute_calls_used = 1
    for step in steps:
        action = step["submission"]["effective_action"]
        messages.append(
            {
                "role": "assistant",
                "content": canonical_response(step["submission"], action=action),
                "trainable": True,
            }
        )
        if action == "execute_code":
            execute_calls_used += 1
            observation = step.get("observation") or {}
            feedback = observation.get("model_feedback")
            if not isinstance(feedback, str):
                raise ValueError(f"{row['task_id']}: execute step has no model feedback")
            messages.append(
                {
                    "role": "tool",
                    "content": feedback_with_budget(
                        feedback, max_execute_calls - execute_calls_used
                    ),
                }
            )
    if any(message["role"] == "tool" and "Private" in message["content"] for message in messages):
        # Revealed counterexamples are allowed, but unrevealed private results must never be serialized.
        if "One failing case has now been revealed" not in "\n".join(
            message["content"] for message in messages if message["role"] == "tool"
        ):
            raise ValueError(f"{row['task_id']}: possible private-test leakage")
    return messages


def stable_key(problem_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare multi-turn Agent SFT smoke data")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/codecontests_plus_repair_v1/repair_api_8b_canonical_33.jsonl",
            "data/processed/codecontests_plus_repair_v1/repair_api_32b_canonical_9.jsonl",
        ],
    )
    parser.add_argument("--train-output", default="data/processed/agent_sft_v3/train.jsonl")
    parser.add_argument("--dev-output", default="data/processed/agent_sft_v3/dev_8.jsonl")
    parser.add_argument("--manifest", default="data/splits/agent_sft_smoke_v3_manifest.json")
    parser.add_argument("--dev-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-train-length", type=int, default=10240)
    parser.add_argument("--max-execute-calls", type=int, default=3)
    args = parser.parse_args()
    project_cache = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(project_cache))

    source_rows = [row for path in args.inputs for row in read_jsonl(path)]
    if len({row["problem_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Agent SFT source contains duplicate problem IDs")
    rows = []
    rejected = []
    for source in source_rows:
        try:
            messages = trajectory_messages(
                source, max_execute_calls=args.max_execute_calls
            )
        except ValueError as error:
            if "execute calls exceed Agent budget" not in str(error):
                raise
            rejected.append(
                {
                    "problem_id": source["problem_id"],
                    "reason": "execution_budget_mismatch",
                    "detail": str(error),
                }
            )
            continue
        rows.append(
            {
                "schema_version": "agent-sft-messages-v3",
                "problem_id": source["problem_id"],
                "task_id": source["task_id"],
                "source": source["source"],
                "teacher_model": source["teacher_model"],
                "messages": messages,
                "metadata": {
                    "trajectory_schema": source["repair_trajectory"]["schema_version"],
                    "normalization": source.get("normalization"),
                },
            }
        )
    for row in rows:
        context_assistant_turns = 0
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            if not message["content"].lstrip().startswith(
                ("<action>execute_code</action>", "<action>final</action>")
            ):
                raise ValueError(f"{row['task_id']}: assistant target lacks canonical action")
            if message.get("trainable") is False:
                context_assistant_turns += 1
            elif message.get("trainable") is not True:
                raise ValueError(
                    f"{row['task_id']}: assistant turn lacks explicit trainable flag"
                )
        if context_assistant_turns != 1:
            raise ValueError(
                f"{row['task_id']}: expected one context-only initial assistant turn"
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    for row in rows:
        encoded = encode_agent_sft_row(row, tokenizer, args.max_length)
        row["token_counts"] = {
            "total": encoded["length"],
            "assistant": sum(label != -100 for label in encoded["labels"]),
            "context": sum(label == -100 for label in encoded["labels"]),
        }
    ordered = sorted(rows, key=lambda row: stable_key(row["problem_id"], args.seed))
    if not 0 < args.dev_size < len(ordered):
        raise ValueError("dev-size must leave non-empty train and dev splits")
    dev, train_candidates = ordered[: args.dev_size], ordered[args.dev_size :]
    excluded_train = [
        row for row in train_candidates if row["token_counts"]["total"] > args.max_train_length
    ]
    train = [
        row for row in train_candidates if row["token_counts"]["total"] <= args.max_train_length
    ]
    if not train:
        raise ValueError("max-train-length removed every training sample")
    write_jsonl(args.train_output, train)
    write_jsonl(args.dev_output, dev)

    def digest(path: str) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "schema_version": "agent-sft-smoke-v3",
        "source_files": args.inputs,
        "tokenizer": {"name_or_path": args.tokenizer, "revision": args.revision},
        "max_length": args.max_length,
        "max_train_length": args.max_train_length,
        "max_execute_calls": args.max_execute_calls,
        "seed": args.seed,
        "problem_ids": {
            "train": [row["problem_id"] for row in train],
            "dev": [row["problem_id"] for row in dev],
        },
        "counts": {
            "train": len(train),
            "dev": len(dev),
            "excluded_train_over_length": len(excluded_train),
            "rejected_before_split": len(rejected),
            "assistant_turns": sum(
                message["role"] == "assistant"
                for row in train + dev
                for message in row["messages"]
            ),
            "trainable_assistant_turns": sum(
                message["role"] == "assistant" and message.get("trainable") is True
                for row in train + dev
                for message in row["messages"]
            ),
            "context_assistant_turns": sum(
                message["role"] == "assistant" and message.get("trainable") is False
                for row in train + dev
                for message in row["messages"]
            ),
            "tokens": {
                split: {
                    key: sum(row["token_counts"][key] for row in values)
                    for key in ("total", "assistant", "context")
                }
                for split, values in (("train", train), ("dev", dev))
            },
        },
        "excluded_train": [
            {
                "problem_id": row["problem_id"],
                "total_tokens": row["token_counts"]["total"],
                "reason": "training_memory_limit",
            }
            for row in excluded_train
        ],
        "rejected": rejected,
        "sha256": {"train": digest(args.train_output), "dev": digest(args.dev_output)},
    }
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"Manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
