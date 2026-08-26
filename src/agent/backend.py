from __future__ import annotations

from dataclasses import replace
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


def full_gate_problem(problem: AgentProblem) -> AgentProblem:
    """Return a view whose private gate contains every current feedback/private test."""

    return replace(problem, hidden_tests=problem.visible_tests + problem.hidden_tests)


def reveal_hidden_counterexample(
    backend: ExecutionBackend,
    code: str,
    problem: AgentProblem,
    hidden: HiddenEvaluation,
    *,
    executions_remaining: int,
    max_feedback_bytes: int,
    min_private_tests: int,
) -> tuple[AgentProblem, ExecutionObservation] | None:
    """Move one failing private case into feedback tests and execute it visibly."""

    if hidden.success or len(problem.hidden_tests) <= min_private_tests:
        return None
    failure_index = next(
        (index for index, case in enumerate(hidden.judge.cases) if not case.passed), None
    )
    if failure_index is None or failure_index >= len(problem.hidden_tests):
        return None
    revealed = problem.hidden_tests[failure_index]
    remaining = problem.hidden_tests[:failure_index] + problem.hidden_tests[failure_index + 1 :]
    updated = replace(
        problem,
        visible_tests=problem.visible_tests + (revealed,),
        hidden_tests=remaining,
    )
    observation = backend.execute_visible(
        code,
        updated,
        executions_remaining=executions_remaining,
        max_feedback_bytes=max_feedback_bytes,
    )
    prefix = (
        "Private validation found a counterexample after the existing feedback tests passed. "
        "One failing case has now been revealed and is part of subsequent feedback tests.\n"
    )
    feedback = prefix + observation.model_feedback
    encoded = feedback.encode("utf-8")
    if len(encoded) > max_feedback_bytes:
        feedback = encoded[:max_feedback_bytes].decode("utf-8", errors="ignore")
    return updated, replace(
        observation,
        model_feedback=feedback,
        revealed_counterexample=True,
        revealed_private_index=failure_index,
        private_tests_remaining=len(remaining),
    )
