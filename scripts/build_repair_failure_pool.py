from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.agent_eval import read_jsonl, write_jsonl


def task_id(problem_id: str, model: str, code: str) -> str:
    return hashlib.sha256(f"{problem_id}\0{model}\0{code}".encode()).hexdigest()


def excluded_ids(paths: list[str]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        for ids in manifest.get("problem_ids", {}).values():
            values.update(ids)
    return values


def has_obvious_repetition(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    return any(count >= 5 for count in collections.Counter(lines).values())


def build_one_shot_pool(
    problems: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    *,
    teacher_model: str,
    excluded: set[str],
    max_response_tokens: int,
) -> list[dict[str, Any]]:
    by_id = {row["problem_id"]: row for row in problems}
    output = []
    seen: set[str] = set()
    for generation in generations:
        problem_id = generation.get("problem_id")
        code = generation.get("code")
        judge = generation.get("judge") or {}
        response = generation.get("response")
        success = judge.get("total", 0) and judge.get("passed") == judge.get("total")
        if (
            not success
            or problem_id in excluded
            or problem_id not in by_id
            or problem_id in seen
            or not isinstance(code, str)
            or not isinstance(response, str)
            or generation.get("finish_reason") != "stop"
            or int(generation.get("response_tokens") or max_response_tokens + 1)
            > max_response_tokens
            or has_obvious_repetition(response)
        ):
            continue
        seen.add(problem_id)
        output.append(
            {
                "schema_version": "verified-one-shot-candidate-v1",
                "problem_id": problem_id,
                "teacher_model": teacher_model,
                "problem": by_id[problem_id]["problem"],
                "difficulty": by_id[problem_id].get("difficulty"),
                "response": response,
                "code": code,
                "generation_tokens": generation.get("response_tokens"),
                "finish_reason": generation.get("finish_reason"),
                "judge": judge,
            }
        )
    return output


def build_pool(
    problems: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    *,
    producer_model: str,
    excluded: set[str],
) -> list[dict[str, Any]]:
    by_id = {row["problem_id"]: row for row in problems}
    output = []
    seen: set[str] = set()
    for generation in generations:
        problem_id = generation.get("problem_id")
        code = generation.get("code")
        judge = generation.get("judge") or {}
        if problem_id in excluded or problem_id not in by_id or not isinstance(code, str) or not code:
            continue
        if judge.get("total", 0) and judge.get("passed") == judge.get("total"):
            continue
        identifier = task_id(problem_id, producer_model, code)
        if identifier in seen:
            continue
        row = by_id[problem_id]
        if not row.get("visible_tests") or not row.get("hidden_tests"):
            raise ValueError(f"{problem_id}: repair problems require visible_tests and hidden_tests")
        seen.add(identifier)
        output.append(
            {
                "task_id": identifier,
                "problem": row,
                "initial_submission": {
                    "producer_model": producer_model,
                    "sample_index": generation.get("sample_index", 0),
                    "response": generation.get("response", ""),
                    "code": code,
                    "finish_reason": generation.get("finish_reason"),
                    "generation_tokens": generation.get("response_tokens"),
                    "source_judge": judge,
                },
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a verifier-ready repair failure pool")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--generations", required=True, nargs="+")
    parser.add_argument("--producer-model", required=True)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--one-shot-output")
    parser.add_argument("--one-shot-max-response-tokens", type=int, default=4096)
    args = parser.parse_args()
    problems = read_jsonl(args.problems)
    generations = [row for path in args.generations for row in read_jsonl(path)]
    rows = build_pool(
        problems,
        generations,
        producer_model=args.producer_model,
        excluded=excluded_ids(args.exclude_manifest),
    )
    write_jsonl(args.output, rows)
    report = {"repair_tasks": len(rows), "repair_output": args.output}
    if args.one_shot_output:
        one_shot = build_one_shot_pool(
            problems,
            generations,
            teacher_model=args.producer_model,
            excluded=excluded_ids(args.exclude_manifest),
            max_response_tokens=args.one_shot_max_response_tokens,
        )
        write_jsonl(args.one_shot_output, one_shot)
        report.update({"one_shot_candidates": len(one_shot), "one_shot_output": args.one_shot_output})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
