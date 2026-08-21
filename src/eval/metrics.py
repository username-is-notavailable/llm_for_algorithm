from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from src.eval.pass_at_k import estimate_pass_at_k


def _compute_metrics(records: Iterable[dict[str, Any]], *, include_difficulty: bool) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot compute metrics for an empty result set")

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem_id"]].append(row)

    extracted = sum(bool(row["extraction_success"]) for row in rows)
    compiled = sum(bool(row.get("judge") and row["judge"]["compiled"]) for row in rows)
    passed_tests = sum(int(row["judge"]["passed"]) for row in rows if row.get("judge"))
    total_tests = sum(int(row["total_tests"]) for row in rows)
    pass_at_1_values: list[float] = []
    pass_at_k_values: list[float] = []
    sample_count = None
    for problem_rows in by_problem.values():
        ordered = sorted(problem_rows, key=lambda row: row["sample_index"])
        correct = sum(
            bool(row.get("judge") and row["judge"]["passed"] == row["judge"]["total"])
            for row in ordered
        )
        if sample_count is None:
            sample_count = len(ordered)
        elif sample_count != len(ordered):
            raise ValueError("Every problem must have the same number of samples")
        pass_at_1_values.append(estimate_pass_at_k(len(ordered), correct, 1))
        pass_at_k_values.append(estimate_pass_at_k(len(ordered), correct, len(ordered)))

    assert sample_count is not None
    problem_count = len(by_problem)
    generation_count = len(rows)
    metrics: dict[str, Any] = {
        "problems": problem_count,
        "generations": generation_count,
        "samples_per_problem": sample_count,
        "code_extraction_success_rate": extracted / generation_count,
        "compile_rate": compiled / generation_count,
        "test_pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "pass@1": sum(pass_at_1_values) / problem_count,
        "average_response_length": sum(int(row["response_length"]) for row in rows)
        / generation_count,
    }
    token_counts = [int(row["response_tokens"]) for row in rows if row.get("response_tokens") is not None]
    if len(token_counts) == generation_count:
        metrics["average_response_tokens"] = sum(token_counts) / generation_count
    finish_reasons = Counter(str(row.get("finish_reason") or "unknown") for row in rows)
    if set(finish_reasons) != {"unknown"}:
        metrics["finish_reasons"] = dict(finish_reasons)
    if sample_count > 1:
        metrics[f"pass@{sample_count}"] = sum(pass_at_k_values) / problem_count
    if include_difficulty:
        difficulties = sorted({str(row["difficulty"]) for row in rows if row.get("difficulty")})
        if difficulties:
            metrics["by_difficulty"] = {
                difficulty: _compute_metrics(
                    [row for row in rows if row.get("difficulty") == difficulty], include_difficulty=False
                )
                for difficulty in difficulties
            }
    return metrics


def compute_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return _compute_metrics(records, include_difficulty=True)
