from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent import (
    ActionParseStatus,
    ActionType,
    AgentConfig,
    AgentProblem,
    ExecutionObservation,
    HiddenEvaluation,
    TerminationReason,
    compute_agent_metrics,
    run_agent,
)
from src.inference.generate import GeneratedText
from src.verifier import JudgeResult, TestCase as JudgeTestCase


WRONG = """<action>execute_code</action>
```cpp
int main() { return 1; }
```"""
RIGHT_EXECUTE = """<action>execute_code</action>
```cpp
int main() { return 0; }
```"""
RIGHT_FINAL = """<action>final</action>
```cpp
int main() { return 0; }
```"""


class FakeGenerator:
    def __init__(self, responses: list[str], token_counts: list[int] | None = None) -> None:
        self.responses = list(responses)
        self.token_counts = list(token_counts or [10] * len(responses))
        self.messages_seen: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        del generation
        self.messages_seen.append([dict(message) for message in messages])
        return GeneratedText(
            text=self.responses.pop(0),
            token_count=self.token_counts.pop(0),
            finish_reason="stop",
        )


class NoCodeGenerator:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason

    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        del messages, generation
        return GeneratedText(text="No program yet.", token_count=3, finish_reason=self.finish_reason)


def _judge(success: bool) -> JudgeResult:
    return JudgeResult(
        compiled=True,
        passed=int(success),
        total=1,
        pass_rate=float(success),
        runtime_error=False,
        timeout=False,
        error_type=None if success else "wrong_answer",
    )


@dataclass
class FakeBackend:
    visible_success_by_code: dict[str, bool]
    hidden_success_by_code: dict[str, bool]

    def execute_visible(
        self, code: str, problem: AgentProblem, *, executions_remaining: int, max_feedback_bytes: int
    ) -> ExecutionObservation:
        del problem, max_feedback_bytes
        result = _judge(self.visible_success_by_code.get(code, False))
        return ExecutionObservation(
            status="passed_visible_tests" if result.pass_rate == 1 else "wrong_answer",
            judge=result,
            model_feedback=f"feedback; remaining={executions_remaining}",
            executions_remaining=executions_remaining,
        )

    def evaluate_hidden(self, code: str, problem: AgentProblem) -> HiddenEvaluation:
        del problem
        return HiddenEvaluation(_judge(self.hidden_success_by_code.get(code, False)))


class BrokenBackend(FakeBackend):
    def evaluate_hidden(self, code: str, problem: AgentProblem) -> HiddenEvaluation:
        del code, problem
        raise RuntimeError("sandbox unavailable")


def _problem(difficulty: str = "easy") -> AgentProblem:
    tests = (JudgeTestCase(input="", output=""),)
    return AgentProblem(
        problem_id="toy:agent",
        problem="Return zero.",
        difficulty=difficulty,
        visible_tests=tests,
        hidden_tests=tests,
    )


def _code(response: str) -> str:
    return response.split("```cpp\n", 1)[1].split("\n```", 1)[0]


def test_agent_repairs_after_feedback_and_explicitly_finalizes() -> None:
    wrong_code = _code(WRONG)
    right_code = _code(RIGHT_EXECUTE)
    generator = FakeGenerator([WRONG, RIGHT_EXECUTE, RIGHT_FINAL])
    backend = FakeBackend(
        visible_success_by_code={right_code: True},
        hidden_success_by_code={right_code: True, wrong_code: False},
    )
    trajectory = run_agent(
        trajectory_id="trajectory-1",
        problem=_problem(),
        model={"name_or_path": "fake"},
        config=AgentConfig(),
        generator=generator,
        backend=backend,
    )
    assert trajectory.termination_reason == TerminationReason.SUCCESS
    assert trajectory.execute_calls == 2
    assert trajectory.candidate_submissions == 3
    assert not trajectory.first_attempt_success
    assert trajectory.final_success
    assert trajectory.repaired
    assert generator.messages_seen[1][-1]["role"] == "tool"
    assert "feedback" in generator.messages_seen[1][-1]["content"]
    assert all("hidden" not in str(messages).lower() for messages in generator.messages_seen)
    assert json.loads(json.dumps(trajectory.to_dict()))["outcome"]["repaired"] is True


def test_fourth_execute_is_recorded_as_auto_final_without_visible_feedback() -> None:
    responses = [
        WRONG,
        WRONG.replace("return 1", "return 2"),
        WRONG.replace("return 1", "return 3"),
        RIGHT_EXECUTE,
    ]
    right_code = _code(RIGHT_EXECUTE)
    trajectory = run_agent(
        trajectory_id="trajectory-budget",
        problem=_problem(),
        model={},
        config=AgentConfig(stop_on_repeated_code=False),
        generator=FakeGenerator(responses),
        backend=FakeBackend({}, {right_code: True}),
    )
    assert trajectory.execute_calls == 3
    assert trajectory.candidate_submissions == 4
    assert trajectory.steps[-1].submission.requested_action == "execute_code"
    assert trajectory.steps[-1].submission.effective_action == ActionType.FINAL
    assert trajectory.steps[-1].observation is None
    assert trajectory.termination_reason == TerminationReason.EXECUTION_BUDGET_EXHAUSTED_AUTO_FINAL
    assert trajectory.final_success


def test_missing_action_falls_back_and_repeated_execute_terminates() -> None:
    no_action = WRONG.replace("<action>execute_code</action>\n", "")
    code = _code(no_action)
    trajectory = run_agent(
        trajectory_id="trajectory-repeat",
        problem=_problem(),
        model={},
        config=AgentConfig(),
        generator=FakeGenerator([no_action, no_action]),
        backend=FakeBackend({}, {code: False}),
    )
    assert trajectory.steps[0].submission.action_parse_status == ActionParseStatus.MISSING_ACTION_FALLBACK
    assert trajectory.termination_reason == TerminationReason.REPEATED_CODE
    assert trajectory.execute_calls == 1


def test_token_budget_turns_candidate_into_final() -> None:
    code = _code(WRONG)
    trajectory = run_agent(
        trajectory_id="trajectory-token",
        problem=_problem(),
        model={},
        config=AgentConfig(max_total_generation_tokens=5),
        generator=FakeGenerator([WRONG], [5]),
        backend=FakeBackend({}, {code: False}),
    )
    assert trajectory.steps[0].submission.effective_action == ActionType.FINAL
    assert trajectory.termination_reason == TerminationReason.TOKEN_BUDGET_EXHAUSTED_AUTO_FINAL


def test_explicit_incorrect_final_and_sandbox_failure_are_distinct() -> None:
    code = _code(RIGHT_FINAL)
    incorrect = run_agent(
        trajectory_id="incorrect-final",
        problem=_problem(),
        model={},
        config=AgentConfig(),
        generator=FakeGenerator([RIGHT_FINAL]),
        backend=FakeBackend({}, {code: False}),
    )
    broken = run_agent(
        trajectory_id="broken-sandbox",
        problem=_problem(),
        model={},
        config=AgentConfig(),
        generator=FakeGenerator([RIGHT_FINAL]),
        backend=BrokenBackend({}, {}),
    )
    assert incorrect.termination_reason == TerminationReason.FINAL_INCORRECT
    assert broken.termination_reason == TerminationReason.SANDBOX_ERROR


def test_no_code_termination_distinguishes_stop_from_length() -> None:
    stopped = run_agent(
        trajectory_id="stopped",
        problem=_problem(),
        model={},
        config=AgentConfig(),
        generator=NoCodeGenerator("stop"),
        backend=FakeBackend({}, {}),
    )
    capped = run_agent(
        trajectory_id="capped",
        problem=_problem(),
        model={},
        config=AgentConfig(),
        generator=NoCodeGenerator("length"),
        backend=FakeBackend({}, {}),
    )
    assert stopped.termination_reason == TerminationReason.MODEL_STOP_WITHOUT_CODE
    assert capped.termination_reason == TerminationReason.CODE_EXTRACTION_FAILED


def test_agent_metrics_separate_first_attempt_and_repair_success() -> None:
    right_code = _code(RIGHT_FINAL)
    direct = run_agent(
        trajectory_id="direct",
        problem=_problem("easy"),
        model={},
        config=AgentConfig(),
        generator=FakeGenerator([RIGHT_FINAL]),
        backend=FakeBackend({}, {right_code: True}),
    )
    wrong_code = _code(WRONG)
    repaired = run_agent(
        trajectory_id="repaired",
        problem=_problem("hard"),
        model={},
        config=AgentConfig(),
        generator=FakeGenerator([WRONG, RIGHT_FINAL]),
        backend=FakeBackend({}, {wrong_code: False, right_code: True}),
    )
    metrics = compute_agent_metrics([direct, repaired])
    assert metrics["first_attempt_success_rate"] == 0.5
    assert metrics["agent_success_rate"] == 1.0
    assert metrics["repair_success_rate"] == 1.0
    assert metrics["success_gain"] == 0.5
    assert metrics["final_hidden_test_pass_rate"] == 1.0
    assert metrics["valid_action_rate"] == 1.0
    assert metrics["explicit_final_rate"] == 1.0
    assert metrics["by_difficulty"]["hard"]["repair_success_rate"] == 1.0
