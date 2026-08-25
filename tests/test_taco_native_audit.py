from scripts.audit_taco_native_solutions import parse_solutions, stable_sample


def test_parse_native_solutions() -> None:
    assert parse_solutions('["print(1)", ""]') == ["print(1)"]
    assert parse_solutions("bad json") == []


def test_native_sample_is_order_independent() -> None:
    rows = [{"problem_id": str(index)} for index in range(10)]
    assert stable_sample(rows, size=4, seed=3) == stable_sample(
        list(reversed(rows)), size=4, seed=3
    )
