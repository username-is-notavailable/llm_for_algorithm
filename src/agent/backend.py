from __future__ import annotations

from typing import Protocol

from src.agent.feedback import format_execution_feedback, observation_status
from src.agent.schemas import AgentProblem, ExecutionObservation, HiddenEvaluation
from src.verifier import judge


class ExecutionBackend(Protocol):
    def execute_visible(
        self, code: str, problem: AgentProblem, *, executions_remaining: int, max_feedback_bytes: int
    ) -> ExecutionObservation: ...

    def evaluate_hidden(self, code: str, problem: AgentProblem) -> HiddenEvaluation: ...


def _judge(code: str, problem: AgentProblem, *, hidden: bool):
    limits = problem.limits
    return judge(
        code,
        problem.hidden_tests if hidden else problem.visible_tests,
        compile_timeout_seconds=limits.compile_timeout_seconds,
        execution_timeout_seconds=limits.execution_timeout_seconds,
        memory_limit_bytes=limits.memory_limit_bytes,
        output_limit_bytes=limits.output_limit_bytes,
    )


class LocalVerifierBackend:
    """Development backend for trusted code; this is not a strong sandbox."""

    def execute_visible(
        self, code: str, problem: AgentProblem, *, executions_remaining: int, max_feedback_bytes: int
    ) -> ExecutionObservation:
        result = _judge(code, problem, hidden=False)
        first_failure_index = next(
            (index for index, case in enumerate(result.cases) if not case.passed), None
        )
        feedback = format_execution_feedback(
            result,
            executions_remaining=executions_remaining,
            max_bytes=max_feedback_bytes,
            first_failing_input=(
                problem.visible_tests[first_failure_index].input
                if first_failure_index is not None
                else None
            ),
        )
        return ExecutionObservation(
            status=observation_status(result),
            judge=result,
            model_feedback=feedback,
            executions_remaining=executions_remaining,
        )

    def evaluate_hidden(self, code: str, problem: AgentProblem) -> HiddenEvaluation:
        return HiddenEvaluation(judge=_judge(code, problem, hidden=True))
