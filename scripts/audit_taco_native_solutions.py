from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.data.agent_eval import read_jsonl
from src.verifier.executor import execute_binary
from src.verifier.judge import _outputs_match


def stable_sample(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['problem_id']}".encode()).digest(),
    )[:size]


def parse_solutions(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [solution for solution in value if isinstance(solution, str) and solution.strip()]


def judge_python(
    code: str,
    tests: list[dict[str, str]],
    *,
    timeout_seconds: float,
    memory_limit_bytes: int,
    output_limit_bytes: int,
    stop_on_first_failure: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qwen3-taco-python-") as directory:
        workdir = Path(directory)
        program = workdir / "solution.py"
        program.write_text("#!/usr/bin/env python3\n" + code, encoding="utf-8")
        program.chmod(program.stat().st_mode | stat.S_IXUSR)
        passed = 0
        first_error = None
        for test in tests:
            result = execute_binary(
                program,
                test["input"],
                workdir,
                timeout_seconds=timeout_seconds,
                memory_limit_bytes=memory_limit_bytes,
                output_limit_bytes=output_limit_bytes,
            )
            if result.timed_out:
                error = "timeout"
            elif result.output_limit_exceeded:
                error = "output_limit"
            elif result.runtime_error:
                error = "runtime_error"
            elif not _outputs_match(result.stdout, test["output"]):
                error = "wrong_answer"
            else:
                error = None
            if error is None:
                passed += 1
            elif first_error is None:
                first_error = error
            if error is not None and stop_on_first_failure:
                break
        return {
            "passed": passed,
            "total": len(tests),
            "pass_rate": passed / len(tests),
            "error_type": first_error,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit TACO native Python solutions on M10 tests")
    parser.add_argument("--dataset", default="data/processed/repair_sft_v1/train_agent_pilot.jsonl")
    parser.add_argument(
        "--source-audit", default="data/processed/repair_sft_v1/source_audit/audit_report.json"
    )
    parser.add_argument(
        "--output", default="data/processed/repair_sft_v1/source_audit/taco_native_sample.json"
    )
    parser.add_argument("--sample-per-group", type=int, default=10)
    parser.add_argument("--max-solutions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--revision", default="d593ed0a2becbbc952230bb89be09189bf1056dc")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    source_audit = json.loads(Path(args.source_audit).read_text(encoding="utf-8"))["problems"]
    passed = [
        row
        for row in rows
        if (source_audit[row["problem_id"]].get("reference_judge") or {}).get("pass_rate") == 1.0
    ]
    failed = [row for row in rows if row not in passed]
    selected = stable_sample(passed, size=args.sample_per_group, seed=args.seed) + stable_sample(
        failed, size=args.sample_per_group, seed=args.seed
    )
    by_index = {int(row["metadata"]["test_source_index"]): row for row in selected}

    from datasets import load_dataset

    data_file = f"hf://datasets/BAAI/TACO@{args.revision}/ALL/train-*.parquet"
    stream = load_dataset("parquet", data_files=data_file, split="train", streaming=True)
    iterator = iter(stream)
    native: dict[int, dict[str, Any]] = {}
    try:
        for index, row in enumerate(iterator):
            if index in by_index:
                native[index] = row
            if len(native) == len(by_index):
                break
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()

    results = []
    for position, row in enumerate(selected, 1):
        index = int(row["metadata"]["test_source_index"])
        taco = native[index]
        solutions = parse_solutions(taco.get("solutions"))
        attempts = []
        for solution in solutions[: args.max_solutions]:
            result = judge_python(
                solution,
                row["tests"],
                timeout_seconds=6,
                memory_limit_bytes=512 * 1024 * 1024,
                output_limit_bytes=1024 * 1024,
                stop_on_first_failure=False,
            )
            attempts.append(result)
            if result["pass_rate"] == 1.0:
                break
        reference = source_audit[row["problem_id"]]["reference_judge"]
        results.append(
            {
                "problem_id": row["problem_id"],
                "test_source_index": index,
                "ocr2_reference_full_pass": reference["pass_rate"] == 1.0,
                "ocr2_reference_pass_rate": reference["pass_rate"],
                "question_exact_match": taco["question"].strip() == row["problem"].strip(),
                "native_solution_count": len(solutions),
                "native_solutions_attempted": len(attempts),
                "native_any_full_pass": any(attempt["pass_rate"] == 1.0 for attempt in attempts),
                "native_attempts": attempts,
            }
        )
        print(f"Audited native solution {position}/{len(selected)}", flush=True)

    groups = {}
    for reference_pass in (True, False):
        group = [row for row in results if row["ocr2_reference_full_pass"] == reference_pass]
        groups["ocr2_pass" if reference_pass else "ocr2_fail"] = {
            "sampled": len(group),
            "question_exact_match": sum(row["question_exact_match"] for row in group),
            "with_native_solution": sum(row["native_solution_count"] > 0 for row in group),
            "native_any_full_pass": sum(row["native_any_full_pass"] for row in group),
        }
    report = {
        "schema_version": "taco-native-solution-audit-v1",
        "taco_revision": args.revision,
        "sample_per_group": args.sample_per_group,
        "max_solutions_per_problem": args.max_solutions,
        "groups": groups,
        "problems": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(groups, indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
