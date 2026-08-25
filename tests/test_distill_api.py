from __future__ import annotations

import hashlib
import json
from typing import Any

from scripts.generate_distillation_api import load_frozen_rows
from src.data.agent_eval import row_test_hash
from src.data.distill_api import distillation_prompt, process_distillation_task
from src.inference.generate import GeneratedText


CORRECT = """<analysis>Add the two values in constant time.</analysis>
```cpp
#include <iostream>
int main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<'\\n';}
```"""


class FakeTeacher:
    def __init__(self, text: str = CORRECT, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason

    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        assert "under 300 words" in messages[0]["content"]
        assert generation["max_new_tokens"] == 8192
        return GeneratedText(
            text=self.text,
            token_count=42,
            finish_reason=self.finish_reason,
            reasoning_content="private reasoning",
            provider_metadata={"request_id": "request-1"},
        )


def config() -> dict[str, Any]:
    return {
        "generation": {"max_new_tokens": 8192},
        "acceptance": {"max_visible_tokens_estimated": 4096},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }


def problem() -> dict[str, Any]:
    return {
        "problem_id": "toy:add",
        "problem": "Read two integers and print their sum.",
        "difficulty": "easy",
        "source": {"dataset": "toy"},
        "visible_tests": [{"input": "1 2\n", "output": "3\n"}],
        "hidden_tests": [{"input": "10 20\n", "output": "30\n"}],
    }


def test_prompt_requires_concise_complete_code() -> None:
    prompt = distillation_prompt("Add two values.", escalated=True)
    assert "under 300 words" in prompt
    assert "exactly one complete GNU C++17 program" in prompt
    assert "smaller teacher" in prompt
    assert "Do not use tool/action tags" in prompt


def test_verified_distillation_separates_private_reasoning() -> None:
    result, accepted = process_distillation_task(
        problem(),
        config(),
        {"stage": "qwen3-8b", "model": "qwen3-8b"},
        FakeTeacher(),
        escalated=False,
    )
    assert accepted
    assert result["judge"]["passed"] == 2
    assert "cases" not in result["judge"]
    assert result["reasoning_content"] == "private reasoning"
    assert "private reasoning" not in result["response"]


def test_unverified_code_is_routed_to_escalation() -> None:
    wrong = CORRECT.replace("a+b", "a-b")
    result, accepted = process_distillation_task(
        problem(),
        config(),
        {"stage": "qwen3-8b", "model": "qwen3-8b"},
        FakeTeacher(wrong),
        escalated=False,
    )
    assert not accepted
    assert result["rejection_reason"] == "verification_failed"


def test_response_without_code_is_rejected_without_judging() -> None:
    result, accepted = process_distillation_task(
        problem(),
        config(),
        {"stage": "qwen3-32b", "model": "qwen3-32b"},
        FakeTeacher("I am still thinking."),
        escalated=True,
    )
    assert not accepted
    assert result["rejection_reason"] == "no_code"
    assert "judge" not in result


def test_load_frozen_rows_validates_dataset_and_test_hashes(tmp_path) -> None:
    row = problem()
    row["tests"] = row["hidden_tests"]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "problem_ids": {"train": [row["problem_id"]]},
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "test_sha256": {row["problem_id"]: row_test_hash(row)},
            }
        ),
        encoding="utf-8",
    )
    assert load_frozen_rows(
        {"dataset": str(dataset), "manifest": str(manifest), "manifest_split": "train"}
    ) == [row]

    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        load_frozen_rows(
            {"dataset": str(dataset), "manifest": str(manifest), "manifest_split": "train"}
        )
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("Modified frozen dataset should be rejected")
