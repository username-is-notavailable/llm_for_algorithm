import pytest

from src.eval.metrics import compute_metrics
from src.eval.pass_at_k import estimate_pass_at_k


def _row(problem_id: str, sample: int, *, extracted: bool, compiled: bool, passed: int) -> dict:
    return {
        "problem_id": problem_id,
        "sample_index": sample,
        "response_length": 10,
        "total_tests": 1,
        "extraction_success": extracted,
        "judge": {"compiled": compiled, "passed": passed, "total": 1} if extracted else None,
    }


def test_pass_at_k_estimator() -> None:
    assert estimate_pass_at_k(1, 1, 1) == 1.0
    assert estimate_pass_at_k(2, 1, 1) == 0.5
    assert estimate_pass_at_k(2, 1, 2) == 1.0
    with pytest.raises(ValueError):
        estimate_pass_at_k(1, 2, 1)


def test_compute_metrics_with_multiple_samples() -> None:
    records = [
        _row("a", 0, extracted=True, compiled=True, passed=1),
        _row("a", 1, extracted=True, compiled=False, passed=0),
        _row("b", 0, extracted=False, compiled=False, passed=0),
        _row("b", 1, extracted=True, compiled=True, passed=0),
    ]
    metrics = compute_metrics(records)
    assert metrics["problems"] == 2
    assert metrics["generations"] == 4
    assert metrics["code_extraction_success_rate"] == 0.75
    assert metrics["compile_rate"] == 0.5
    assert metrics["test_pass_rate"] == 0.25
    assert metrics["pass@1"] == 0.25
    assert metrics["pass@2"] == 0.5
    assert metrics["average_response_length"] == 10
