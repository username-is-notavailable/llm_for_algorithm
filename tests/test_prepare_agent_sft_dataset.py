from __future__ import annotations

from src.agent.schemas import AgentConfig, AgentProblem
from src.verifier import TestCase as JudgeTestCase

import scripts.prepare_agent_sft_dataset as prepare


CODE = "```cpp\nint main() { return 0; }\n```"


def _row(actions: list[str]) -> dict:
    return {
        "task_id": "repair:test",
        "initial_submission": {"code": "int main() { return 1; }"},
        "initial_observation": {"model_feedback": "wrong answer"},
        "repair_trajectory": {
            "steps": [
                {
                    "submission": {
                        "effective_action": action,
                        "response": f"<action>{action}</action>\n{CODE}",
                    },
                    "observation": (
                        {"model_feedback": "- Executions remaining: 2"}
                        if action == "execute_code"
                        else None
                    ),
                }
                for action in actions
            ]
        },
    }


def _patch_problem(monkeypatch) -> None:
    tests = (JudgeTestCase(input="", output=""),)
    problem = AgentProblem(
        problem_id="test", problem="Return zero.", visible_tests=tests, hidden_tests=tests
    )
    monkeypatch.setattr(prepare, "load_problem", lambda *_: problem)
    monkeypatch.setattr(
        prepare,
        "build_initial_messages",
        lambda *_: [{"role": "system", "content": "protocol"}],
    )


def test_final_after_execute_reuses_code_without_repeating_it(monkeypatch) -> None:
    _patch_problem(monkeypatch)
    messages = prepare.repair_messages(
        _row(["execute_code", "final"]),
        {},
        agent_config=AgentConfig(),
        verifier_config={},
    )
    assistant = [message["content"] for message in messages if message["role"] == "assistant"]
    assert assistant[-1] == "<action>final</action>"


def test_direct_final_keeps_complete_code(monkeypatch) -> None:
    _patch_problem(monkeypatch)
    messages = prepare.repair_messages(
        _row(["final"]),
        {},
        agent_config=AgentConfig(),
        verifier_config={},
    )
    assert messages[-1]["content"] == f"<action>final</action>\n{CODE}"


def test_rollout_aligned_repair_uses_failed_execution_as_masked_context(monkeypatch) -> None:
    _patch_problem(monkeypatch)
    messages = prepare.rollout_aligned_repair_messages(
        _row(["execute_code", "final"]),
        {},
        agent_config=AgentConfig(),
        verifier_config={},
    )
    assert messages[1]["role"] == "assistant"
    assert messages[1]["trainable"] is False
    assert "return 1" in messages[1]["content"]
    assert messages[2]["role"] == "tool"
    assert "Executions remaining: 2" in messages[2]["content"]
    assert messages[-1]["content"] == "<action>final</action>"


def test_rollout_aligned_direct_final_keeps_corrected_code(monkeypatch) -> None:
    _patch_problem(monkeypatch)
    messages = prepare.rollout_aligned_repair_messages(
        _row(["final"]),
        {},
        agent_config=AgentConfig(),
        verifier_config={},
    )
    assert messages[-1]["content"] == f"<action>final</action>\n{CODE}"


def test_rollout_aligned_repair_enforces_total_execution_budget(monkeypatch) -> None:
    _patch_problem(monkeypatch)
    import pytest

    with pytest.raises(ValueError, match="exceeds execution budget"):
        prepare.rollout_aligned_repair_messages(
            _row(["execute_code", "execute_code", "execute_code", "final"]),
            {},
            agent_config=AgentConfig(),
            verifier_config={},
        )
