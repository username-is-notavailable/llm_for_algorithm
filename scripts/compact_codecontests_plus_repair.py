from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def judge_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    return {
        key: value.get(key)
        for key in ("compiled", "passed", "total", "pass_rate", "runtime_error", "timeout", "error_type")
        if key in value
    }


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(handle, row: dict[str, Any]) -> tuple[int, int]:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    offset = handle.tell()
    handle.write(payload)
    return offset, len(payload)


def reconstruct_tests(problem: dict[str, Any]) -> list[dict[str, Any]]:
    split = problem["agent_test_split"]
    visible = dict(zip(split["visible_indices"], problem["visible_tests"], strict=True))
    hidden = dict(zip(split["hidden_indices"], problem["hidden_tests"], strict=True))
    indexed = {**visible, **hidden}
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError(f"{problem['problem_id']}: test split indices are not contiguous")
    return [indexed[index] for index in range(len(indexed))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact checker-backed repair data without loading it into RAM")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--failure-pool", required=True)
    parser.add_argument("--one-shot-seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    source_problems = Path(args.problems)
    source_failures = Path(args.failure_pool)
    source_one_shots = Path(args.one_shot_seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems_path = output_dir / "problems.jsonl"
    failures_path = output_dir / "failure_pool.jsonl"
    one_shots_path = output_dir / "one_shot_seeds.jsonl"
    index_path = output_dir / "problems.index.json"

    records: dict[str, list[int]] = {}
    counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    problem_iter = iter_jsonl(source_problems)
    failure_iter = iter_jsonl(source_failures)
    one_shot_iter = iter_jsonl(source_one_shots)
    with problems_path.open("wb") as problem_handle, failures_path.open("wb") as failure_handle, one_shots_path.open("wb") as one_shot_handle:
        for position, triple in enumerate(zip(problem_iter, failure_iter, one_shot_iter, strict=True), 1):
            problem, failure, one_shot = triple
            embedded = failure["problem"]
            problem_id = problem["problem_id"]
            if embedded["problem_id"] != problem_id or one_shot["problem_id"] != problem_id:
                raise ValueError(f"Row {position}: problem IDs are not aligned")
            for key in ("problem", "visible_tests", "hidden_tests", "metadata"):
                if problem.get(key) != embedded.get(key):
                    raise ValueError(f"{problem_id}: embedded problem differs in {key}")

            compact_problem = dict(problem)
            # `tests` is exactly reconstructable from the frozen visible/hidden split.
            if compact_problem.get("tests") != reconstruct_tests(compact_problem):
                raise ValueError(f"{problem_id}: frozen split does not reconstruct all tests")
            compact_problem.pop("tests", None)
            offset, length = write_record(problem_handle, compact_problem)
            records[problem_id] = [offset, length]

            initial = dict(failure["initial_submission"])
            initial["source_judge"] = judge_summary(initial.get("source_judge"))
            compact_failure = {
                "schema_version": "checker-backed-repair-task-v2",
                "task_id": failure["task_id"],
                "problem_id": problem_id,
                "initial_submission": initial,
            }
            write_record(failure_handle, compact_failure)

            compact_one_shot = dict(one_shot)
            compact_one_shot["schema_version"] = "checker-backed-one-shot-seed-v2"
            compact_one_shot["source_judge"] = judge_summary(one_shot.get("source_judge"))
            write_record(one_shot_handle, compact_one_shot)
            counts["problems"] += 1
            counts["visible_tests"] += len(problem["visible_tests"])
            counts["hidden_tests"] += len(problem["hidden_tests"])
            error_types[str(initial["source_judge"].get("error_type"))] += 1
            if position % 25 == 0:
                print(f"Compacted {position} rows", flush=True)

    index_payload = {"schema_version": "jsonl-byte-offset-index-v1", "records": records}
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    files = [problems_path, failures_path, one_shots_path, index_path]
    manifest = {
        "schema_version": "codecontests-plus-repair-compact-v2",
        "counts": {**counts, "initial_error_types": dict(error_types)},
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        },
        "source_files": {
            "problems": str(source_problems),
            "failure_pool": str(source_failures),
            "one_shot_seeds": str(source_one_shots),
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
