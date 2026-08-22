from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.eval.evaluator import configure_shard, evaluate, load_problems, validate_split_manifest
from src.inference.generate import GeneratedText
from src.inference.prompts import build_code_prompt
from src.eval.merge_shards import main as merge_shards


ADD_CODE = """```cpp
#include <iostream>
int main() { long long a, b; std::cin >> a >> b; std::cout << a + b << '\\n'; }
```"""
PARITY_CODE = """<think>Use modulo two.</think>
```cpp
#include <iostream>
int main() { long long n; std::cin >> n; std::cout << (n % 2 ? "ODD" : "EVEN") << '\\n'; }
```"""


class FakeGenerator:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = list(responses)

    def generate_batch(
        self, prompts: list[str], *, num_samples: int, generation: dict[str, Any]
    ) -> list[list[str]]:
        del generation
        results = [self.responses.pop(0) for _ in prompts]
        assert all(len(result) == num_samples for result in results)
        return [
            [GeneratedText(text=value, token_count=len(value), finish_reason="stop") for value in result]
            for result in results
        ]


def _write_dataset(path: Path) -> None:
    problems = [
        {
            "problem_id": "toy:add",
            "problem": "Add two integers.",
            "language": "cpp",
            "tests": [{"input": "1 2\n", "output": "3\n"}],
        },
        {
            "problem_id": "toy:parity",
            "problem": "Print EVEN or ODD.",
            "language": "cpp",
            "tests": [{"input": "3\n", "output": "ODD\n"}],
        },
    ]
    path.write_text("".join(json.dumps(value) + "\n" for value in problems), encoding="utf-8")


def test_prompt_uses_output_protocol_v1() -> None:
    prompt = build_code_prompt("Add two integers.")
    assert "<think>" in prompt
    assert "```cpp" in prompt
    assert "<answer>" not in prompt
    assert "Do not put fenced code blocks" not in prompt


def test_default_toy_dataset_has_seven_unique_problems() -> None:
    problems = load_problems("data/fixtures/eval_toy_v1.jsonl")
    assert len(problems) == 7
    assert len({problem["problem_id"] for problem in problems}) == 7


def test_evaluator_writes_reproducible_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.jsonl"
    _write_dataset(dataset_path)
    config = {
        "experiment": {"name": "test-eval", "output_dir": str(tmp_path / "outputs"), "seed": 42},
        "model": {"name_or_path": "fake"},
        "inference": {"request_batch_size": 2},
        "prompt": {"template": "output_protocol_v1"},
        "dataset": {"path": str(dataset_path)},
        "generation": {"num_samples": 1, "max_new_tokens": 64, "do_sample": False},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }
    output_dir, metrics = evaluate(config, FakeGenerator([[ADD_CODE], [PARITY_CODE]]))
    assert metrics["pass@1"] == 1.0
    assert metrics["compile_rate"] == 1.0
    assert metrics["test_pass_rate"] == 1.0
    assert metrics["finish_reasons"] == {"stop": 2}
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "environment.json").is_file()
    assert (output_dir / "generations.jsonl").is_file()
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")) == metrics
    assert yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8")) == config
    assert len((output_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_evaluator_resumes_completed_problems(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.jsonl"
    _write_dataset(dataset_path)
    config = {
        "experiment": {"name": "resume-eval", "output_dir": str(tmp_path / "outputs"), "seed": 42},
        "model": {"name_or_path": "fake"},
        "inference": {"request_batch_size": 1},
        "prompt": {"template": "output_protocol_v1"},
        "dataset": {"path": str(dataset_path)},
        "generation": {"num_samples": 1, "max_new_tokens": 64, "do_sample": False},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }
    output_dir, _ = evaluate(config, FakeGenerator([[ADD_CODE], [PARITY_CODE]]))
    generations = output_dir / "generations.jsonl"
    first_record = generations.read_text(encoding="utf-8").splitlines()[0]
    generations.write_text(first_record + "\n", encoding="utf-8")
    (output_dir / "metrics.json").unlink()

    _, metrics = evaluate(config, FakeGenerator([[PARITY_CODE]]), resume=output_dir)

    assert metrics["pass@1"] == 1.0
    assert len(generations.read_text(encoding="utf-8").splitlines()) == 2


def test_evaluator_deterministically_shards_problems(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.jsonl"
    _write_dataset(dataset_path)
    base = {
        "experiment": {"name": "shard-eval", "output_dir": str(tmp_path / "outputs"), "seed": 42},
        "model": {"name_or_path": "fake"},
        "inference": {"request_batch_size": 1},
        "prompt": {"template": "output_protocol_v1"},
        "dataset": {"path": str(dataset_path)},
        "generation": {"num_samples": 1, "max_new_tokens": 64, "do_sample": False},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }
    first_dir, _ = evaluate(configure_shard(base, 0, 2), FakeGenerator([[ADD_CODE]]))
    second_dir, _ = evaluate(configure_shard(base, 1, 2), FakeGenerator([[PARITY_CODE]]))
    first = json.loads((first_dir / "generations.jsonl").read_text())
    second = json.loads((second_dir / "generations.jsonl").read_text())
    assert first["problem_id"] == "toy:add"
    assert second["problem_id"] == "toy:parity"
    assert configure_shard(base, 0, 2)["experiment"]["name"].endswith("shard-01-of-02")
    assert "shard" not in base["dataset"]


def test_merge_eval_shards_validates_and_restores_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "toy.jsonl"
    _write_dataset(dataset_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    base = {
        "experiment": {"name": "merge-eval", "output_dir": str(tmp_path / "outputs"), "seed": 42},
        "model": {"name_or_path": str(model_path.resolve())},
        "inference": {"request_batch_size": 1},
        "prompt": {"template": "output_protocol_v1"},
        "dataset": {"path": str(dataset_path)},
        "generation": {"num_samples": 1, "max_new_tokens": 64, "do_sample": False},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }
    first_dir, _ = evaluate(configure_shard(base, 0, 2), FakeGenerator([[ADD_CODE]]))
    second_dir, _ = evaluate(configure_shard(base, 1, 2), FakeGenerator([[PARITY_CODE]]))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    merged = tmp_path / "merged"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_eval_shards.py",
            "--config",
            str(config_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(merged),
            str(first_dir),
            str(second_dir),
        ],
    )
    assert merge_shards() == 0
    rows = [json.loads(line) for line in (merged / "generations.jsonl").read_text().splitlines()]
    assert [row["problem_id"] for row in rows] == ["toy:add", "toy:parity"]
    assert json.loads((merged / "metrics.json").read_text())["pass@1"] == 1.0


def test_dataset_rejects_duplicate_problem_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    value = {
        "problem_id": "same",
        "problem": "x",
        "language": "cpp",
        "tests": [{"input": "", "output": ""}],
    }
    path.write_text(json.dumps(value) + "\n" + json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate problem_id"):
        load_problems(path)


def test_frozen_manifest_rejects_changed_problem_order(tmp_path: Path) -> None:
    problems = [{"problem_id": "a"}, {"problem_id": "b"}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"problem_ids": {"smoke": ["b", "a"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen manifest"):
        validate_split_manifest(problems, manifest, "smoke")
