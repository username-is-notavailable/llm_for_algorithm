from __future__ import annotations

import argparse
import collections
import copy
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


def _full_success_at_step(step: dict[str, Any]) -> bool:
    observation = step.get("observation") or {}
    visible = observation.get("judge") or {}
    hidden = (step.get("hidden_evaluation") or {}).get("judge") or {}
    return bool(
        visible.get("total")
        and visible.get("passed") == visible.get("total")
        and hidden.get("total")
        and hidden.get("passed") == hidden.get("total")
    )


def _merged_success_evaluation(step: dict[str, Any]) -> dict[str, Any]:
    visible = step["observation"]["judge"]
    hidden = step["hidden_evaluation"]["judge"]
    cases = copy.deepcopy(visible.get("cases", [])) + copy.deepcopy(hidden.get("cases", []))
    total = int(visible["total"]) + int(hidden["total"])
    return {
        "judge": {
            "compiled": True,
            "passed": total,
            "total": total,
            "pass_rate": 1.0,
            "runtime_error": False,
            "timeout": False,
            "error_type": None,
            "compile_stderr": "",
            "cases": cases,
        }
    }


def recover_intermediate_success(row: dict[str, Any], *, source_run: str) -> dict[str, Any] | None:
    """Truncate post-success regression and append a provenance-marked deterministic final."""

    value = json.loads(json.dumps(row))
    trajectory = value.get("repair_trajectory") or {}
    steps = trajectory.get("steps") or []
    success_index = next((index for index, step in enumerate(steps) if _full_success_at_step(step)), None)
    if success_index is None or success_index == len(steps) - 1:
        return None
    success = steps[success_index]
    submission = success["submission"]
    next_step = steps[success_index + 1]
    final_evaluation = _merged_success_evaluation(success)
    final_turn = int(success["turn"]) + 1
    synthetic_final = {
        "turn": final_turn,
        "prompt_messages": copy.deepcopy(next_step["prompt_messages"]),
        "submission": {
            "turn": final_turn,
            "response": _canonical_response(
                f"```cpp\n{submission['code'].strip()}\n```", "final"
            ),
            "code": submission["code"],
            "code_sha256": submission["code_sha256"],
            "requested_action": "final",
            "effective_action": "final",
            "action_parse_status": "explicit",
            "prompt_tokens": None,
            "generation_tokens": 0,
            "finish_reason": "normalized",
            "reasoning_content": None,
            "provider_metadata": {
                "normalization": "reuse_full-pass_execute_as_final",
                "source_turn": int(success["turn"]),
            },
        },
        "observation": None,
        "hidden_evaluation": copy.deepcopy(final_evaluation),
        "previous_visible_pass_rate": 1.0,
        "current_visible_pass_rate": None,
        "delta_visible_pass_rate": None,
    }
    truncated_turns = [int(step["turn"]) for step in steps[success_index + 1 :]]
    trajectory["steps"] = steps[: success_index + 1] + [synthetic_final]
    trajectory["hidden_evaluation"] = final_evaluation
    trajectory["termination_reason"] = "success"
    outcome = trajectory["outcome"]
    first = trajectory["steps"][0]
    first_success = _full_success_at_step(first) or bool(
        first.get("observation") is None
        and ((first.get("hidden_evaluation") or {}).get("judge") or {}).get("pass_rate") == 1
    )
    outcome.update(
        {
            "first_attempt_success": first_success,
            "final_success": True,
            "repaired": not first_success,
            "execute_calls": sum(step.get("observation") is not None for step in trajectory["steps"]),
            "candidate_submissions": len(trajectory["steps"]),
            "total_generation_tokens": sum(
                int(step["submission"].get("generation_tokens") or 0)
                for step in trajectory["steps"]
            ),
            "termination_reason": "success",
        }
    )
    value["accepted"] = True
    value["rejection_reason"] = None
    value["normalization"] = {
        "schema_version": "intermediate-success-recovery-v1",
        "source_run": source_run,
        "source_success_turn": int(success["turn"]),
        "truncated_turns": truncated_turns,
        "method": "truncate_after_full-pass_execute_and_append_identical_final",
        "execution_results_changed": False,
        "code_changed": False,
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
    recovered = [
        value
        for row in rows
        if not (row.get("repair_trajectory", {}).get("outcome") or {}).get("final_success")
        and (value := recover_intermediate_success(row, source_run=str(run))) is not None
    ]
    canonical.extend(recovered)
    recovered_ids = {row["task_id"] for row in recovered}
    failed = [
        row
        for row in rows
        if row.get("rejection_reason") == "repair_failed_full_tests"
        and row["task_id"] not in recovered_ids
    ]
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
        "canonicalized_protocol_rows": sum(
            bool(row["normalization"].get("changed_turns")) for row in canonical
        ),
        "recovered_intermediate_successes": len(recovered),
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
