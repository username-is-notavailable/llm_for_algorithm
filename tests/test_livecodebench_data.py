from __future__ import annotations

import base64
import json
import pickle
import zlib

import pytest

from src.data.leakage import find_leaks
from src.data.livecodebench import adapt_problem, split_manifest, stratified_splits


def _row(question_id: str = "100_A", difficulty: str = "easy") -> dict:
    private = [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(private)))).decode()
    return {
        "question_title": "Add",
        "question_content": f"Add two integers. Stable variant {question_id} with enough detail.",
        "platform": "codeforces",
        "question_id": question_id,
        "contest_id": "100",
        "contest_date": "2023-08-21T00:00:00",
        "starter_code": "",
        "difficulty": difficulty,
        "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]),
        "private_test_cases": encoded,
        "metadata": "{}",
    }


def test_adapts_public_and_compressed_private_tests() -> None:
    problem = adapt_problem(_row())
    assert problem["problem_id"] == "livecodebench:codeforces:100_A"
    assert problem["difficulty"] == "easy"
    assert problem["metadata"]["contest_date"] == "2023-08-21T00:00:00"
    assert problem["tests"] == [
        {"input": "1 2\n", "output": "3\n"},
        {"input": "2 3\n", "output": "5\n"},
    ]


def test_rejects_non_stdin_test_case() -> None:
    row = _row()
    row["public_test_cases"] = json.dumps([{"input": [], "output": [], "testtype": "functional"}])
    with pytest.raises(ValueError, match="stdin"):
        adapt_problem(row)


def test_stratified_splits_are_disjoint_and_reproducible() -> None:
    problems = [adapt_problem(_row(f"{difficulty}_{index}", difficulty)) for difficulty in ("easy", "medium", "hard") for index in range(10)]
    first = stratified_splits(problems, seed=7, dev_fraction=0.2, smoke_size=3, max_problems=30)
    second = stratified_splits(problems, seed=7, dev_fraction=0.2, smoke_size=3, max_problems=30)
    assert [[row["problem_id"] for row in first[name]] for name in ("eval", "dev", "smoke")] == [
        [row["problem_id"] for row in second[name]] for name in ("eval", "dev", "smoke")
    ]
    eval_ids = {row["problem_id"] for row in first["eval"]}
    dev_ids = {row["problem_id"] for row in first["dev"]}
    assert len(first["eval"]) == 24
    assert len(first["dev"]) == 6
    assert eval_ids.isdisjoint(dev_ids)
    assert {row["problem_id"] for row in first["smoke"]} <= dev_ids
    manifest = split_manifest(first, dataset="dataset", release="v1", revision="abc", seed=7)
    assert manifest["counts"] == {"dev": 6, "eval": 24, "smoke": 3}


def test_leakage_detects_id_exact_text_and_near_duplicate() -> None:
    long_text = "Compute the maximum sum over every contiguous subarray using dynamic programming and print the result. " * 3
    eval_rows = [{"problem_id": "eval:1", "problem": long_text}]
    training = [
        {"problem_id": "eval:1", "problem": "unrelated"},
        {"problem_id": "train:exact", "problem": long_text.upper()},
        {"problem_id": "train:near", "problem": long_text + " Time limit two seconds."},
    ]
    assert [leak["reason"] for leak in find_leaks(eval_rows, training)] == [
        "problem_id",
        "normalized_sha256",
        "near_duplicate_simhash",
    ]
