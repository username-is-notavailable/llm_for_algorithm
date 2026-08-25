from scripts.audit_taco_native_solutions import judge_python, parse_solutions, stable_sample


def test_parse_native_solutions() -> None:
    assert parse_solutions('["print(1)", ""]') == ["print(1)"]
    assert parse_solutions("bad json") == []


def test_native_sample_is_order_independent() -> None:
    rows = [{"problem_id": str(index)} for index in range(10)]
    assert stable_sample(rows, size=4, seed=3) == stable_sample(
        list(reversed(rows)), size=4, seed=3
    )


def test_python_judge_can_stop_after_first_failure() -> None:
    result = judge_python(
        "print('wrong')",
        [{"input": "", "output": "right"}, {"input": "", "output": "wrong"}],
        timeout_seconds=1,
        memory_limit_bytes=256 * 1024 * 1024,
        output_limit_bytes=65536,
        stop_on_first_failure=True,
    )
    assert result["passed"] == 0
    assert result["total"] == 2
    assert result["error_type"] == "wrong_answer"
