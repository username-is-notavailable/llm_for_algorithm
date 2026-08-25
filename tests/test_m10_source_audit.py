from scripts.audit_m10_sources import parse_tests, tests_sha256 as hash_tests


def test_parse_tests_and_hash_are_deterministic() -> None:
    value = {"inputs": ["1", "1", "2"], "outputs": ["a", "a", "b"]}
    tests = parse_tests(value, max_tests=10)
    assert tests == [{"input": "1", "output": "a"}, {"input": "2", "output": "b"}]
    assert hash_tests(tests) == hash_tests(list(tests))


def test_parse_tests_rejects_function_calling_and_misaligned_lists() -> None:
    assert parse_tests({"fn_name": "f", "inputs": [], "outputs": []}, max_tests=10) is None
    assert parse_tests({"inputs": ["1"], "outputs": []}, max_tests=10) is None
