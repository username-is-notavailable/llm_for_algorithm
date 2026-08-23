from __future__ import annotations

import pytest

from scripts.prepare_sft_ab import derive_variants, render_code_only_response, select_matched_rows


def _row(
    problem_id: str,
    *,
    reasoning_tokens: int = 100,
    response_tokens: int = 200,
    total_tokens: int = 300,
    verified: bool = True,
) -> dict:
    return {
        "problem_id": problem_id,
        "problem": f"Problem {problem_id}",
        "prompt": f"Long prompt {problem_id}",
        "reasoning": "A complete explanation.",
        "code": "int main() { return 0; }",
        "response": "old response",
        "verified": verified,
        "difficulty": "easy",
        "metadata": {"platform": "test"},
        "token_counts": {
            "prompt": 100,
            "reasoning": reasoning_tokens,
            "code": 10,
            "response": response_tokens,
            "total": total_tokens,
        },
    }


def test_selection_filters_limits_and_is_deterministic() -> None:
    rows = [
        _row("a"),
        _row("b"),
        _row("long", reasoning_tokens=101),
        _row("unverified", verified=False),
    ]
    first, eligible = select_matched_rows(
        rows,
        samples=2,
        reasoning_max_tokens=100,
        response_max_tokens=250,
        total_max_tokens=400,
        seed=42,
    )
    second, _ = select_matched_rows(
        rows,
        samples=2,
        reasoning_max_tokens=100,
        response_max_tokens=250,
        total_max_tokens=400,
        seed=42,
    )
    assert eligible == 2
    assert [row["problem_id"] for row in first] == [row["problem_id"] for row in second]


def test_selection_rejects_insufficient_rows() -> None:
    with pytest.raises(ValueError, match="Only 1 eligible"):
        select_matched_rows(
            [_row("a")],
            samples=2,
            reasoning_max_tokens=100,
            response_max_tokens=250,
            total_max_tokens=400,
            seed=42,
        )


def test_variants_share_ids_and_code_only_removes_reasoning() -> None:
    source = _row("same")
    short_rows, code_rows = derive_variants([source], lambda text: len(text.split()))
    short = short_rows[0]
    code = code_rows[0]
    assert short["problem_id"] == code["problem_id"] == "same"
    assert short["response"] == "old response"
    assert short["metadata"]["sft_variant"] == "short_reasoning_v2"
    assert code["reasoning"] == ""
    assert code["response"] == render_code_only_response(source["code"])
    assert "<think>" not in code["prompt"]
    assert code["metadata"]["sft_variant"] == "code_only_v2"
    assert code["token_counts"]["reasoning"] == 0
    assert source["metadata"] == {"platform": "test"}
