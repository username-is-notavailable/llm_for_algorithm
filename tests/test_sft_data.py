from __future__ import annotations

from src.data.sft import (
    adapt_ocr2,
    balanced_order,
    deduplicate_candidates,
    is_eval_leak,
    normalize_difficulty,
    render_response,
    split_generation,
    unsupported_problem_reason,
)
from scripts.prepare_sft import _select_by_token_length

CODE = "#include <iostream>\nint main() { return 0; }"


def _row(question_id: str = "q1", sample_id: str = "s1", pass_rate: str = "1.0") -> dict:
    response = f"<think>Explain the algorithm carefully and prove why it works.</think>\n```cpp\n{CODE}\n```"
    return {
        "id": sample_id,
        "question_id": question_id,
        "r1_generation": response,
        "solution": CODE,
        "judgement": "right",
        "pass_rate": pass_rate,
        "source": "codeforces",
        "license": "cc-by-4.0",
        "dataset": "taco",
        "split": "train",
        "difficulty": "1600",
        "index": "7",
    }


def _filter() -> dict:
    return {
        "require_judgement": "right",
        "minimum_pass_rate": 0.8,
        "allow_unverified": False,
        "minimum_reasoning_characters": 20,
        "maximum_response_characters": 50000,
    }


def test_parses_and_renders_output_protocol_v1() -> None:
    reasoning, code = split_generation(_row()["r1_generation"]) or (None, None)
    assert reasoning == "Explain the algorithm carefully and prove why it works."
    assert code == CODE
    assert split_generation(render_response(reasoning, code)) == (reasoning, code)


def test_adapts_ocr2_and_preserves_provenance() -> None:
    sample = adapt_ocr2(_row(), "A complete programming problem with input and output specifications.")
    assert sample["problem_id"] == "ocr2:q1"
    assert sample["difficulty"] == "medium"
    assert sample["verified"] is True
    assert sample["metadata"]["original_id"] == "s1"
    assert sample["metadata"]["dataset_index"] == "7"


def test_filters_unverified_mismatched_and_duplicate_samples() -> None:
    weaker = _row(sample_id="unverified", pass_rate="-1")
    mismatch = _row(question_id="q2")
    mismatch["solution"] = "int main() {}"
    duplicate = _row(sample_id="duplicate")
    rows, rejected = deduplicate_candidates([weaker, mismatch, _row(), duplicate], _filter())
    assert len(rows) == 1
    assert rejected["unverified"] == 1
    assert rejected["solution_mismatch"] == 1
    assert rejected["duplicate_problem"] == 1


def test_difficulty_normalization_and_balanced_order() -> None:
    assert [normalize_difficulty(value) for value in ("EASY", "3", "2100", "UNKNOWN_DIFFICULTY")] == [
        "easy", "medium", "hard", "unknown"
    ]
    rows = []
    for difficulty, platform in (("easy", "a"), ("easy", "a"), ("hard", "b")):
        row = adapt_ocr2(_row(f"{difficulty}:{platform}:{len(rows)}", str(len(rows))), "A sufficiently detailed problem statement for testing.")
        row["difficulty"] = difficulty
        row["metadata"]["platform"] = platform
        rows.append(row)
    ordered = balanced_order(rows, seed=7)
    assert ordered[0]["difficulty"] != ordered[1]["difficulty"]
    raw_rows = [_row("raw-a"), _row("raw-b")]
    assert {row["question_id"] for row in balanced_order(raw_rows, seed=7)} == {"raw-a", "raw-b"}


def test_eval_leakage_exact_and_near_duplicate() -> None:
    problem = "Find the maximum subarray sum using dynamic programming and output the result. " * 3
    from src.data.leakage import normalized_problem, problem_sha256, problem_simhash

    fingerprints = [{
        "sha256": problem_sha256(problem),
        "simhash": problem_simhash(problem),
        "normalized_length": len(normalized_problem(problem)),
    }]
    assert is_eval_leak(problem.upper(), fingerprints) == "eval_exact"
    assert is_eval_leak(problem + "The time limit is two seconds.", fingerprints) == "eval_near_duplicate"


def test_token_length_filter_rejects_whole_sample_and_refills() -> None:
    class CharacterTokenizer:
        @staticmethod
        def encode(value: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is False
            return list(value)

    rows = [
        adapt_ocr2(_row(f"q{index}", f"s{index}"), "A sufficiently detailed problem statement.")
        for index in range(3)
    ]
    rows[0]["reasoning"] = "x" * 1000
    rows[0]["response"] = render_response(rows[0]["reasoning"], rows[0]["code"])
    from collections import Counter

    rejected = Counter()
    selected = _select_by_token_length(
        rows,
        CharacterTokenizer(),
        target_size=2,
        maximum_total_tokens=600,
        rejected=rejected,
    )

    assert [row["problem_id"] for row in selected] == ["ocr2:q1", "ocr2:q2"]
    assert rejected["total_tokens"] == 1
    assert all(row["token_counts"]["total"] <= 600 for row in selected)


def test_rejects_interactive_and_missing_statement_tasks() -> None:
    assert unsupported_problem_reason("This is an interactive problem. Ask queries and flush output.") == "interactive_problem"
    assert unsupported_problem_reason("Unfortunately someone ate the problem statement. Solve without the statement.") == "missing_problem_statement"
    assert unsupported_problem_reason("Solve a complete stdin and stdout programming problem.") is None
