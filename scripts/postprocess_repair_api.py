from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.agent_eval import read_jsonl, write_jsonl


def _canonical_response(response: str, action: str) -> str:
    return f"<action>{action}</action>\n{response.strip()}"


def canonicalize_success(row: dict[str, Any], *, source_run: str) -> dict[str, Any]:
    value = json.loads(json.dumps(row))
    trajectory = value["repair_trajectory"]
    steps = trajectory["steps"]
    changed_turns: list[int] = []
    for index, step in enumerate(steps):
        submission = step["submission"]
        action = "final" if index == len(steps) - 1 else submission["effective_action"]
        if submission["action_parse_status"] != "explicit" or submission.get(
            "requested_action"
        ) != action:
            submission["response"] = _canonical_response(submission["response"], action)
            submission["requested_action"] = action
            submission["effective_action"] = action
            submission["action_parse_status"] = "explicit"
            changed_turns.append(int(step["turn"]))
    trajectory["termination_reason"] = "success"
    trajectory["outcome"]["termination_reason"] = "success"
    value["accepted"] = True
    value["rejection_reason"] = None
    value["normalization"] = {
        "schema_version": "action-canonicalization-v1",
        "source_run": source_run,
        "changed_turns": changed_turns,
        "method": "prefix_effective_action_and_finalize_verified_code",
        "execution_results_changed": False,
    }
    return value


def _best_failed_step(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for step in row.get("repair_trajectory", {}).get("steps", []):
        submission = step.get("submission") or {}
        if not submission.get("code"):
            continue
        visible_rate = step.get("current_visible_pass_rate")
        if visible_rate is None:
            observation = step.get("observation") or {}
            judge = observation.get("judge") or {}
            total = int(judge.get("total") or 0)
            visible_rate = int(judge.get("passed") or 0) / total if total else -1.0
        candidates.append((float(visible_rate), int(step.get("turn", 0)), step))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def escalation_payload(
    row: dict[str, Any], original: dict[str, Any], *, source_run: str
) -> dict[str, Any] | None:
    best = _best_failed_step(row)
    if best is None:
        return None
    submission = best["submission"]
    parent_task_id = row["task_id"]
    producer_model = str(row["teacher_model"])
    identifier = hashlib.sha256(
        f"{parent_task_id}\0{producer_model}\0{submission['code']}".encode()
    ).hexdigest()
    return {
        "task_id": identifier,
        "problem": original["problem"],
        "initial_submission": {
            "producer_model": producer_model,
            "sample_index": 0,
            "response": submission["response"],
            "code": submission["code"],
            "finish_reason": submission.get("finish_reason"),
            "generation_tokens": submission.get("generation_tokens"),
            "source_judge": (best.get("hidden_evaluation") or {}).get("judge"),
            "parent_task_id": parent_task_id,
            "source_run": source_run,
            "source_turn": best.get("turn"),
            "source_visible_pass_rate": best.get("current_visible_pass_rate"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize successful repairs and export escalation tasks")
    parser.add_argument("--run", required=True)
    parser.add_argument("--failure-pool", required=True)
    parser.add_argument("--canonical-output", required=True)
    parser.add_argument("--escalation-output", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    rows = read_jsonl(run / "accepted.jsonl") + read_jsonl(run / "rejected.jsonl")
    original = {row["task_id"]: row for row in read_jsonl(args.failure_pool)}
    successful = [
        row
        for row in rows
        if (row.get("repair_trajectory", {}).get("outcome") or {}).get("final_success")
    ]
    canonical = [canonicalize_success(row, source_run=str(run)) for row in successful]
    failed = [row for row in rows if row.get("rejection_reason") == "repair_failed_full_tests"]
    escalation = [
        payload
        for row in failed
        if (payload := escalation_payload(row, original[row["task_id"]], source_run=str(run)))
    ]
    write_jsonl(args.canonical_output, canonical)
    write_jsonl(args.escalation_output, escalation)
    report = {
        "input_rows": len(rows),
        "canonical_successes": len(canonical),
        "canonicalized_protocol_rows": sum(bool(row["normalization"]["changed_turns"]) for row in canonical),
        "escalation_tasks": len(escalation),
        "unrecoverable_reasons": dict(
            collections.Counter(
                row.get("rejection_reason")
                for row in rows
                if row not in successful and row.get("rejection_reason") != "repair_failed_full_tests"
            )
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
