import json
from pathlib import Path

from src.data.agent_eval import (
    row_test_hash,
    select_agent_dev,
    split_visible_hidden_tests,
)


def _row(index: int, difficulty: str = "easy") -> dict:
    return {
        "problem_id": f"problem-{index}",
        "problem": "x",
        "difficulty": difficulty,
        "tests": [
            {"input": f"{case}\n", "output": f"{case}\n"} for case in range(10)
        ],
    }


def test_agent_dev_selection_keeps_smoke_prefix_and_is_deterministic() -> None:
    rows = [_row(index, ["easy", "medium", "hard"][index % 3]) for index in range(20)]
    first = select_agent_dev(rows, smoke_ids=["problem-3", "problem-1"], size=11, seed=42)
    second = select_agent_dev(reversed(rows), smoke_ids=["problem-3", "problem-1"], size=11, seed=42)
    assert [row["problem_id"] for row in first] == [row["problem_id"] for row in second]
    assert [row["problem_id"] for row in first[:2]] == ["problem-3", "problem-1"]


def test_visible_hidden_split_is_disjoint_reproducible_and_hashed() -> None:
    first = split_visible_hidden_tests(_row(1), seed=42, visible_fraction=0.2, visible_max=5)
    second = split_visible_hidden_tests(_row(1), seed=42, visible_fraction=0.2, visible_max=5)
    assert first == second
    assert len(first["visible_tests"]) == 2
    assert len(first["hidden_tests"]) == 8
    assert first["tests"] == first["hidden_tests"]
    assert set(first["agent_test_split"]["visible_indices"]).isdisjoint(
        first["agent_test_split"]["hidden_indices"]
    )
    assert row_test_hash(first) == row_test_hash(second)


def test_committed_agent_manifest_matches_generated_data() -> None:
    manifest_path = Path("data/splits/agent_eval_v1_problem_ids.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["smoke"] == 10
    assert manifest["counts"]["dev"] == 60
    assert manifest["problem_ids"]["dev"][:10] == manifest["problem_ids"]["smoke"]
