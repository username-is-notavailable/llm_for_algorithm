from __future__ import annotations

import pytest

from scripts.prepare_sft_short import percentile, select_short_rows


def _row(problem_id: str, response_tokens: int, *, verified: bool = True) -> dict:
    return {
        "problem_id": problem_id,
        "verified": verified,
        "token_counts": {"response": response_tokens},
    }


def test_short_selection_preserves_source_order_and_filters() -> None:
    rows = [_row("long", 101), _row("a", 50), _row("bad", 20, verified=False), _row("b", 75)]
    selected = select_short_rows(rows, response_max_tokens=100, samples=2)
    assert [row["problem_id"] for row in selected] == ["a", "b"]


def test_short_selection_rejects_insufficient_or_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="Only 1"):
        select_short_rows([_row("a", 50)], response_max_tokens=100, samples=2)
    with pytest.raises(ValueError, match="duplicate"):
        select_short_rows([_row("a", 50), _row("a", 60)], response_max_tokens=100, samples=2)


def test_percentile_is_deterministic_nearest_rank_index() -> None:
    assert percentile([5, 1, 3, 2, 4], 0.5) == 3
