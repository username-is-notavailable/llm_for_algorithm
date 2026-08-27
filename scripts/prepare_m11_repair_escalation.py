from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    from scripts.postprocess_repair_api import (
        canonicalize_success,
        escalation_payload,
        recover_intermediate_success,
    )
except ModuleNotFoundError:
    from postprocess_repair_api import (
        canonicalize_success,
        escalation_payload,
        recover_intermediate_success,
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def judge_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    return {
        key: value.get(key)
        for key in ("compiled", "passed", "total", "pass_rate", "runtime_error", "timeout", "error_type")
        if key in value
    }


def compact_submission(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "turn", "response", "code", "code_sha256", "requested_action", "effective_action",
            "action_parse_status", "prompt_tokens", "generation_tokens", "finish_reason",
            "reasoning_content", "provider_metadata",
        )
        if key in value
    }


def compact_success(row: dict[str, Any]) -> dict[str, Any]:
    trajectory = row["repair_trajectory"]
    steps = []
    for step in trajectory["steps"]:
        observation = step.get("observation")
        steps.append({
            "turn": step["turn"],
            "submission": compact_submission(step["submission"]),
            "observation": (
                {
                    "model_feedback": observation.get("model_feedback"),
                    "executions_remaining": observation.get("executions_remaining"),
                    "revealed_counterexample": observation.get("revealed_counterexample", False),
                }
                if observation else None
            ),
            "current_visible_pass_rate": step.get("current_visible_pass_rate"),
            "delta_visible_pass_rate": step.get("delta_visible_pass_rate"),
        })
    initial = dict(row["initial_submission"])
    initial["source_judge"] = judge_summary(initial.get("source_judge"))
    return {
        "schema_version": "repair-example-compact-v3",
        "task_id": row["task_id"],
        "problem_id": row["problem_id"],
        "difficulty": row.get("difficulty"),
        "source": row.get("source"),
        "failure_producer_model": row.get("failure_producer_model"),
        "teacher_model": row.get("teacher_model"),
        "initial_submission": initial,
        "initial_observation": {
            "model_feedback": row["initial_observation"]["model_feedback"],
            "executions_remaining": row["initial_observation"].get("executions_remaining"),
            "revealed_counterexample": row["initial_observation"].get("revealed_counterexample", False),
        },
        "repair_trajectory": {
            "schema_version": trajectory["schema_version"],
            "steps": steps,
            "termination_reason": trajectory.get("termination_reason"),
            "outcome": trajectory.get("outcome"),
        },
        "normalization": row.get("normalization"),
    }


def compact_escalation(
    value: dict[str, Any], problem_id: str, source_row: dict[str, Any]
) -> dict[str, Any]:
    initial = dict(value["initial_submission"])
    initial["source_judge"] = judge_summary(initial.get("source_judge"))
    return {
        "schema_version": "checker-backed-repair-task-v2",
        "task_id": value["task_id"],
        "problem_id": problem_id,
        "initial_submission": initial,
        "metadata": {
            "source_teacher_model": source_row.get("teacher_model"),
            "source_rejection_reason": source_row.get("rejection_reason"),
            "source_termination_reason": (source_row.get("repair_trajectory") or {}).get(
                "termination_reason"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact 8B successes and prepare 32B escalation")
    parser.add_argument("--run", required=True)
    parser.add_argument("--failure-pool", required=True)
    parser.add_argument("--canonical-output", required=True)
    parser.add_argument("--escalation-output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    originals = {row["task_id"]: row for row in iter_jsonl(Path(args.failure_pool))}
    canonical_path = Path(args.canonical_output)
    escalation_path = Path(args.escalation_output)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    escalation_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()

    with canonical_path.open("w", encoding="utf-8") as canonical_handle, escalation_path.open("w", encoding="utf-8") as escalation_handle:
        for status in ("accepted", "rejected"):
            for row in iter_jsonl(run / f"{status}.jsonl"):
                trajectory = row.get("repair_trajectory") or {}
                final_success = bool((trajectory.get("outcome") or {}).get("final_success"))
                if final_success:
                    value = canonicalize_success(row, source_run=str(run))
                    counts["strict_success" if status == "accepted" else "protocol_recovered"] += 1
                    canonical_handle.write(json.dumps(compact_success(value), ensure_ascii=False) + "\n")
                    continue
                recovered = recover_intermediate_success(row, source_run=str(run))
                if recovered is not None:
                    counts["intermediate_success_recovered"] += 1
                    canonical_handle.write(json.dumps(compact_success(recovered), ensure_ascii=False) + "\n")
                    continue
                original = originals[row["task_id"]]
                value = escalation_payload(
                    row,
                    {"problem": {"problem_id": original["problem_id"]}},
                    source_run=str(run),
                )
                if value is None:
                    value = {
                        "task_id": original["task_id"],
                        "initial_submission": original["initial_submission"],
                    }
                    counts["fallback_to_original"] += 1
                escalation_handle.write(
                    json.dumps(
                        compact_escalation(value, original["problem_id"], row),
                        ensure_ascii=False,
                    ) + "\n"
                )
                counts["escalated"] += 1
                rejection_reasons[row.get("rejection_reason") or "unknown"] += 1
                termination_reasons[trajectory.get("termination_reason") or "unknown"] += 1

    manifest = {
        "schema_version": "m11-repair-escalation-v1",
        "source_run": str(run),
        "counts": dict(counts),
        "escalation_rejection_reasons": dict(rejection_reasons),
        "escalation_termination_reasons": dict(termination_reasons),
        "files": {
            "canonical": {"path": str(canonical_path), "bytes": canonical_path.stat().st_size, "sha256": sha256_file(canonical_path)},
            "escalation": {"path": str(escalation_path), "bytes": escalation_path.stat().st_size, "sha256": sha256_file(escalation_path)},
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
