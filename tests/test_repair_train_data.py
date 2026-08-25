import json

from scripts.prepare_repair_train import parse_taco_tests


def test_parse_taco_stdin_tests() -> None:
    value = json.dumps({"inputs": ["1 2\n", "2 3\n"], "outputs": ["3\n", "5\n"]})
    assert parse_taco_tests(value, max_tests=10) == [
        {"input": "1 2\n", "output": "3\n"},
        {"input": "2 3\n", "output": "5\n"},
    ]


def test_parse_taco_rejects_function_call_and_insufficient_tests() -> None:
    assert parse_taco_tests({"fn_name": "solve", "inputs": [1], "outputs": [1]}, max_tests=10) is None
    assert parse_taco_tests({"inputs": ["1"], "outputs": ["1"]}, max_tests=10) is None
