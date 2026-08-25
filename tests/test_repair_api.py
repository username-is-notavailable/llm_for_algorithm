from __future__ import annotations

from typing import Any

from src.data.repair_api import process_task
from src.inference.generate import GeneratedText


FIXED = """<action>execute_code</action>
```cpp
#include <iostream>
int main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<'\\n';}
```"""
FINAL = FIXED.replace("execute_code", "final")


class FakeTeacher:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        del generation
        assert "earlier student" in messages[1]["content"]
        value = FIXED if self.calls == 0 else FINAL
        self.calls += 1
        return GeneratedText(
            value,
            20,
            "stop",
            reasoning_content="fix addition",
            provider_metadata={"request_id": f"r-{self.calls}"},
        )


def test_process_task_accepts_verified_explicit_repair() -> None:
    payload = {
        "task_id": "task-1",
        "problem": {
            "problem_id": "toy:add",
            "problem": "Read two integers and print their sum.",
            "language": "cpp",
            "difficulty": "easy",
            "visible_tests": [{"input": "1 2\n", "output": "3\n"}],
            "hidden_tests": [{"input": "10 20\n", "output": "30\n"}],
        },
        "initial_submission": {
            "student_model": "fake-base",
            "response": "bad",
            "code": "int main(){BROKEN}",
        },
    }
    config = {
        "api": {"model": "qwen3-8b"},
        "generation": {"max_new_tokens": 100},
        "agent": {"max_execute_calls": 1, "max_candidate_submissions": 2},
        "verifier": {
            "compile_timeout_seconds": 5,
            "execution_timeout_seconds": 1,
            "memory_limit_mb": 256,
            "output_limit_bytes": 65536,
        },
    }
    result, accepted = process_task(payload, config, FakeTeacher())
    assert accepted
    assert result["repair_trajectory"]["outcome"]["final_success"]
    first = result["repair_trajectory"]["steps"][0]["submission"]
    assert first["reasoning_content"] == "fix addition"
    assert first["provider_metadata"]["request_id"] == "r-1"
