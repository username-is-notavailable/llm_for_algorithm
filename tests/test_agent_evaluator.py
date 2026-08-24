from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.agent.evaluator import configure_agent_shard, evaluate_agent
from src.data.agent_eval import row_test_hash
from src.inference.generate import GeneratedText


FINAL = """<action>final</action>
```cpp
#include <iostream>
int main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<'\\n';}
```"""


class FakeGenerator:
    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        del messages, generation
        return GeneratedText(FINAL, 20, "stop")


def _config(tmp_path: Path) -> dict[str, Any]:
    rows = []
    for index in range(2):
        row = {
            "problem_id": f"toy:{index}",
            "source": "toy",
            "problem": "Add two integers.",
            "language": "cpp",
            "difficulty": "easy",
            "visible_tests": [{"input": "1 2\n", "output": "3\n"}],
            "hidden_tests": [{"input": "10 20\n", "output": "30\n"}],
            "tests": [{"input": "10 20\n", "output": "30\n"}],
        }
        rows.append(row)
    data = tmp_path / "agent.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "problem_ids": {"dev": [row["problem_id"] for row in rows]},
                "test_sha256": {row["problem_id"]: row_test_hash(row) for row in rows},
            }
        ),
        encoding="utf-8",
    )
    return {
        "experiment": {"name": "agent-test", "output_dir": str(tmp_path / "out"), "seed": 42},
        "model": {"name_or_path": "fake"},
        "prompt": {},
        "dataset": {"path": str(data), "manifest": str(manifest), "manifest_split": "dev"},
        "generation": {"num_samples": 1, "max_new_tokens": 100},
        "agent": {"max_execute_calls": 1, "max_candidate_submissions": 2},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }


def test_agent_evaluator_writes_and_resumes_trajectories(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output, metrics = evaluate_agent(config, FakeGenerator())
    assert metrics["agent_success_rate"] == 1.0
    assert len((output / "trajectories.jsonl").read_text().splitlines()) == 2
    assert yaml.safe_load((output / "config.yaml").read_text()) == config
    _, resumed = evaluate_agent(config, FakeGenerator(), resume=output)
    assert resumed == metrics
    assert len((output / "trajectories.jsonl").read_text().splitlines()) == 2


def test_agent_shard_configuration_is_non_mutating(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shard = configure_agent_shard(config, 1, 2)
    assert shard["dataset"]["shard"] == {"index": 1, "count": 2}
    assert shard["experiment"]["name"].endswith("shard-02-of-02")
    assert "shard" not in config["dataset"]
