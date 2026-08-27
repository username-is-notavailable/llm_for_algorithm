from __future__ import annotations

import pytest

from scripts.prepare_agent_sft_smoke import (
    feedback_with_budget,
    problem_only,
    trajectory_messages,
)


def _feedback(remaining: int = 3) -> str:
    return (
        "Execution result:\n- Status: WRONG_ANSWER\n- Visible tests passed: 0/1\n"
        f"- Executions remaining: {remaining}"
    )


def _source_row(*, teacher_executes: int = 1) -> dict:
    system = {"role": "system", "content": "agent protocol"}
    user = {
        "role": "user",
        "content": (
            "Problem statement"
            "\n\nAn earlier student submitted the following GNU C++17 program:"
            "\n```cpp\nbad\n```\n\nIt received this real execution-environment observation:"
            "\nwrong answer"
        ),
    }
    steps = []
    prompt = [system, user]
    for turn in range(teacher_executes):
        response = f"<action>execute_code</action>\n```cpp\nfix{turn}\n```"
        steps.append(
            {
                "turn": turn,
                "prompt_messages": list(prompt),
                "submission": {
                    "response": response,
                    "effective_action": "execute_code",
                },
                "observation": {"model_feedback": _feedback(3 - turn - 1)},
            }
        )
        prompt.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "tool", "content": _feedback(3 - turn - 1)},
            ]
        )
    steps.append(
        {
            "turn": teacher_executes,
            "prompt_messages": list(prompt),
            "submission": {
                "response": "<action>final</action>\n```cpp\nfixed\n```",
                "effective_action": "final",
            },
            "observation": None,
        }
    )
    return {
        "task_id": "task",
        "initial_submission": {"code": "bad"},
        "initial_observation": {"model_feedback": _feedback()},
        "repair_trajectory": {"steps": steps},
    }


def test_problem_only_removes_bundled_failure_context() -> None:
    prompt = _source_row()["repair_trajectory"]["steps"][0]["prompt_messages"][1]["content"]
    assert problem_only(prompt) == "Problem statement"


def test_feedback_budget_replaces_or_appends_budget() -> None:
    assert feedback_with_budget(_feedback(3), 1).endswith("Executions remaining: 1")
    assert feedback_with_budget("Execution result:\n[truncated after 4096 bytes]", 2).endswith(
        "Executions remaining: 2"
    )


def test_trajectory_matches_agent_loop_and_masks_initial_attempt() -> None:
    messages = trajectory_messages(_source_row())
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assistants = [message for message in messages if message["role"] == "assistant"]
    assert [message["trainable"] for message in assistants] == [False, True, True]
    tools = [message for message in messages if message["role"] == "tool"]
    assert tools[0]["content"].endswith("Executions remaining: 2")
    assert tools[1]["content"].endswith("Executions remaining: 1")


def test_trajectory_rejects_more_execute_calls_than_agent_budget() -> None:
    with pytest.raises(ValueError, match="execute calls exceed Agent budget"):
        trajectory_messages(_source_row(teacher_executes=3), max_execute_calls=3)
