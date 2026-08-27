from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.agent.controller import build_initial_messages
from src.agent.schemas import AgentConfig
from src.data.agent_eval import read_jsonl, write_jsonl
from src.data.problem_store import IndexedProblemStore
from src.data.repair_api import load_problem, repair_prompt
from src.training.sft import encode_agent_sft_row
from src.utils.config import load_config, require_sections


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(problem_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()


def canonical_response(submission: dict[str, Any]) -> str:
    action = submission["effective_action"]
    response = submission["response"].strip()
    prefix = f"<action>{action}</action>"
    return response if response.startswith(prefix) else f"{prefix}\n{response}"


def feedback_with_budget(feedback: str, remaining: int) -> str:
    import re

    updated, count = re.subn(
        r"(?m)^- Executions remaining: \d+$",
        f"- Executions remaining: {remaining}",
        feedback,
    )
    if count == 0:
        return f"{feedback.rstrip()}\n- Executions remaining: {remaining}"
    if count != 1:
        raise ValueError("Execution feedback contains multiple remaining-budget lines")
    return updated


def repair_messages(
    row: dict[str, Any],
    problem_row: dict[str, Any],
    *,
    agent_config: AgentConfig,
    verifier_config: dict[str, Any],
) -> list[dict[str, Any]]:
    problem = load_problem(problem_row, verifier_config)
    initial_code = row["initial_submission"]["code"].strip()
    initial_feedback = row["initial_observation"]["model_feedback"]
    repair_problem = type(problem)(
        problem_id=problem.problem_id,
        problem=repair_prompt(problem.problem, initial_code, initial_feedback),
        visible_tests=problem.visible_tests,
        hidden_tests=problem.hidden_tests,
        difficulty=problem.difficulty,
        source=problem.source,
        limits=problem.limits,
        metadata=problem.metadata,
    )
    messages: list[dict[str, Any]] = build_initial_messages(repair_problem, agent_config)
    execute_calls = 0
    steps = row["repair_trajectory"]["steps"]
    if not steps or steps[-1]["submission"]["effective_action"] != "final":
        raise ValueError(f"{row['task_id']}: repair trajectory must end in final")
    for step in steps:
        submission = step["submission"]
        action = submission["effective_action"]
        final_reuses_last_code = action == "final" and execute_calls > 0
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "<action>final</action>"
                    if final_reuses_last_code
                    else canonical_response(submission)
                ),
                "trainable": True,
            }
        )
        if action == "execute_code":
            execute_calls += 1
            if execute_calls > agent_config.max_execute_calls:
                raise ValueError(f"{row['task_id']}: repair exceeds execution budget")
            observation = step.get("observation") or {}
            feedback = observation.get("model_feedback")
            if not isinstance(feedback, str):
                raise ValueError(f"{row['task_id']}: execute step has no feedback")
            messages.append(
                {
                    "role": "tool",
                    "content": feedback_with_budget(
                        feedback, agent_config.max_execute_calls - execute_calls
                    ),
                }
            )
    return messages


def rollout_aligned_repair_messages(
    row: dict[str, Any],
    problem_row: dict[str, Any],
    *,
    agent_config: AgentConfig,
    verifier_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render repair supervision in the same state layout used by online rollout."""

    problem = load_problem(problem_row, verifier_config)
    messages: list[dict[str, Any]] = build_initial_messages(problem, agent_config)
    initial_code = row["initial_submission"]["code"].strip()
    initial_feedback = row["initial_observation"]["model_feedback"]
    messages.extend(
        [
            {
                "role": "assistant",
                "content": (
                    "<action>execute_code</action>\n"
                    f"```cpp\n{initial_code}\n```"
                ),
                # Preserve the failed action as rollout context without teaching
                # the model to reproduce the known-wrong program.
                "trainable": False,
            },
            {
                "role": "tool",
                "content": feedback_with_budget(
                    initial_feedback, agent_config.max_execute_calls - 1
                ),
            },
        ]
    )

    teacher_execute_calls = 0
    steps = row["repair_trajectory"]["steps"]
    if not steps or steps[-1]["submission"]["effective_action"] != "final":
        raise ValueError(f"{row['task_id']}: repair trajectory must end in final")
    for step in steps:
        submission = step["submission"]
        action = submission["effective_action"]
        final_reuses_teacher_code = action == "final" and teacher_execute_calls > 0
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "<action>final</action>"
                    if final_reuses_teacher_code
                    else canonical_response(submission)
                ),
                "trainable": True,
            }
        )
        if action == "execute_code":
            teacher_execute_calls += 1
            total_execute_calls = 1 + teacher_execute_calls
            if total_execute_calls > agent_config.max_execute_calls:
                raise ValueError(f"{row['task_id']}: aligned repair exceeds execution budget")
            observation = step.get("observation") or {}
            feedback = observation.get("model_feedback")
            if not isinstance(feedback, str):
                raise ValueError(f"{row['task_id']}: execute step has no feedback")
            messages.append(
                {
                    "role": "tool",
                    "content": feedback_with_budget(
                        feedback, agent_config.max_execute_calls - total_execute_calls
                    ),
                }
            )
    return messages


def token_stats(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen M11 Agent SFT dataset")
    parser.add_argument("--config", default="configs/data/m11_agent_sft_v3.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "input", "output", "split", "tokenizer", "agent", "verifier")
    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parents[1] / "cache" / "huggingface")
    )

    inputs = config["input"]
    store = IndexedProblemStore(inputs["problem_dataset"], inputs["problem_index"])
    one_shot = list(read_jsonl(inputs["one_shot"]))
    repair_sources = [
        row for path in inputs["repair"] for row in read_jsonl(path)
    ]
    repair_layout = inputs.get("repair_layout", "embedded_failure_v1")
    if repair_layout not in {"embedded_failure_v1", "rollout_aligned_v1"}:
        raise ValueError(f"Unsupported repair_layout: {repair_layout}")
    repair_ids = [row["problem_id"] for row in repair_sources]
    if len(set(repair_ids)) != len(repair_ids):
        raise ValueError("Repair sources contain duplicate problem IDs")
    agent_config = AgentConfig(
        max_execute_calls=int(config["agent"]["max_execute_calls"]),
        max_candidate_submissions=int(config["agent"]["max_execute_calls"]) + 1,
        max_feedback_bytes=int(config["agent"]["max_feedback_bytes"]),
    )

    rows = []
    for row in one_shot:
        value = dict(row)
        value["metadata"] = {**value.get("metadata", {}), "trajectory_family": "one_shot"}
        rows.append(value)
    budget_excluded = []
    for source in repair_sources:
        teacher_execute_calls = sum(
            step["submission"]["effective_action"] == "execute_code"
            for step in source["repair_trajectory"]["steps"]
        )
        if (
            repair_layout == "rollout_aligned_v1"
            and 1 + teacher_execute_calls > agent_config.max_execute_calls
        ):
            budget_excluded.append(
                {
                    "problem_id": source["problem_id"],
                    "task_id": source["task_id"],
                    "teacher_execute_calls": teacher_execute_calls,
                }
            )
            continue
        message_builder = (
            rollout_aligned_repair_messages
            if repair_layout == "rollout_aligned_v1"
            else repair_messages
        )
        rows.append(
            {
                "schema_version": (
                    "agent-sft-messages-v6"
                    if repair_layout == "rollout_aligned_v1"
                    else "agent-sft-messages-v5"
                ),
                "problem_id": source["problem_id"],
                "task_id": source["task_id"],
                "source": source.get("source"),
                "teacher_model": source.get("teacher_model"),
                "messages": message_builder(
                    source,
                    store.get(source["problem_id"]),
                    agent_config=agent_config,
                    verifier_config=config["verifier"],
                ),
                "metadata": {
                    "trajectory_family": "repair",
                    "failure_producer_model": source.get("failure_producer_model"),
                    "normalization": source.get("normalization"),
                    "repair_layout": repair_layout,
                    "final_reuses_last_executed_code": any(
                        step["submission"]["effective_action"] == "execute_code"
                        for step in source["repair_trajectory"]["steps"]
                    ),
                },
            }
        )

    from transformers import AutoTokenizer

    tokenizer_config = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["name_or_path"], revision=tokenizer_config["revision"]
    )
    max_length = int(tokenizer_config["max_length"])
    accepted, over_length = [], []
    for position, row in enumerate(rows, 1):
        try:
            encoded = encode_agent_sft_row(row, tokenizer, max_length)
        except ValueError as error:
            if "max_length=" not in str(error):
                raise
            over_length.append({"problem_id": row["problem_id"], "detail": str(error)})
            continue
        row["token_counts"] = {
            "total": encoded["length"],
            "assistant": sum(label != -100 for label in encoded["labels"]),
            "context": sum(label == -100 for label in encoded["labels"]),
        }
        accepted.append(row)
        if position % 100 == 0:
            print(f"Tokenized {position}/{len(rows)}", flush=True)

    seed = int(config["split"]["seed"])
    all_problem_ids = sorted({row["problem_id"] for row in accepted}, key=lambda x: stable_key(x, seed))
    dev_problem_count = int(config["split"]["dev_problems"])
    if not 0 < dev_problem_count < len(all_problem_ids):
        raise ValueError("dev_problems must leave non-empty train and dev problem sets")
    dev_ids = set(all_problem_ids[:dev_problem_count])
    dev = [row for row in accepted if row["problem_id"] in dev_ids]
    train = [row for row in accepted if row["problem_id"] not in dev_ids]
    train.sort(key=lambda row: stable_key(f"{row['problem_id']}:{row['task_id']}", seed))
    dev.sort(key=lambda row: stable_key(f"{row['problem_id']}:{row['task_id']}", seed))

    output = config["output"]
    write_jsonl(output["train"], train)
    write_jsonl(output["dev"], dev)
    family_counts = {
        split: dict(Counter(row["metadata"]["trajectory_family"] for row in values))
        for split, values in (("all", accepted), ("train", train), ("dev", dev))
    }
    manifest = {
        "schema_version": (
            "agent-sft-v3-manifest"
            if repair_layout == "rollout_aligned_v1"
            else "agent-sft-v2-manifest"
        ),
        "config": config,
        "source_sha256": {
            "one_shot": sha256_file(inputs["one_shot"]),
            "repair": {path: sha256_file(path) for path in inputs["repair"]},
        },
        "counts": {
            "source_rows": len(rows),
            "accepted": len(accepted),
            "train": len(train),
            "dev": len(dev),
            "unique_problems": len(all_problem_ids),
            "over_length": len(over_length),
            "budget_excluded": len(budget_excluded),
            "families": family_counts,
        },
        "token_counts": {
            split: token_stats(row["token_counts"]["total"] for row in values)
            for split, values in (("all", accepted), ("train", train), ("dev", dev))
        },
        "dev_problem_ids": sorted(dev_ids),
        "over_length": over_length,
        "budget_excluded": budget_excluded,
        "sha256": {"train": sha256_file(output["train"]), "dev": sha256_file(output["dev"])},
    }
    manifest_path = Path(output["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "token_counts": manifest["token_counts"]}, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
