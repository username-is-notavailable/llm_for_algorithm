from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from src.agent.schemas import AgentTrajectory
from src.agent.schemas import ActionParseStatus, ActionType


def _compute(trajectories: list[AgentTrajectory], *, difficulty: bool) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("Cannot compute agent metrics for an empty trajectory set")
    first_successes = sum(row.first_attempt_success for row in trajectories)
    final_successes = sum(row.final_success for row in trajectories)
    first_failures = len(trajectories) - first_successes
    repaired = sum(row.repaired for row in trajectories)
    execute_calls = sum(row.execute_calls for row in trajectories)
    generation_tokens = sum(row.total_generation_tokens for row in trajectories)
    submissions = [step.submission for row in trajectories for step in row.steps]
    explicit_actions = sum(
        submission.action_parse_status == ActionParseStatus.EXPLICIT for submission in submissions
    )
    explicit_finals = sum(
        submission.action_parse_status == ActionParseStatus.EXPLICIT
        and submission.effective_action == ActionType.FINAL
        for submission in submissions
    )
    final_hidden_passed = sum(
        row.hidden_evaluation.judge.passed for row in trajectories if row.hidden_evaluation
    )
    final_hidden_total = sum(
        row.hidden_evaluation.judge.total for row in trajectories if row.hidden_evaluation
    )
    successful = [row for row in trajectories if row.final_success]
    max_submissions = max(row.candidate_submissions for row in trajectories)
    cumulative_success = {
        str(limit): sum(
            any(
                step.hidden_evaluation and step.hidden_evaluation.success
                for step in row.steps[:limit]
            )
            for row in trajectories
        )
        / len(trajectories)
        for limit in range(1, max_submissions + 1)
    }
    result: dict[str, Any] = {
        "trajectories": len(trajectories),
        "first_attempt_success_rate": first_successes / len(trajectories),
        "agent_success_rate": final_successes / len(trajectories),
        "repair_success_rate": repaired / first_failures if first_failures else 0.0,
        "success_gain": (final_successes - first_successes) / len(trajectories),
        "final_hidden_test_pass_rate": (
            final_hidden_passed / final_hidden_total if final_hidden_total else 0.0
        ),
        "valid_action_rate": explicit_actions / len(submissions) if submissions else 0.0,
        "action_fallback_rate": (
            (len(submissions) - explicit_actions) / len(submissions) if submissions else 0.0
        ),
        "explicit_final_rate": explicit_finals / len(trajectories),
        "average_execute_calls": execute_calls / len(trajectories),
        "average_candidate_submissions": sum(row.candidate_submissions for row in trajectories)
        / len(trajectories),
        "average_generation_tokens": generation_tokens / len(trajectories),
        "tokens_per_success": generation_tokens / final_successes if final_successes else None,
        "average_execute_calls_on_success": (
            sum(row.execute_calls for row in successful) / len(successful) if successful else None
        ),
        "termination_reasons": dict(
            Counter(str(row.termination_reason) for row in trajectories)
        ),
        "success_by_candidate_submission": cumulative_success,
    }
    if difficulty:
        labels = sorted({row.difficulty for row in trajectories if row.difficulty})
        if labels:
            result["by_difficulty"] = {
                label: _compute(
                    [row for row in trajectories if row.difficulty == label], difficulty=False
                )
                for label in labels
            }
    return result


def compute_agent_metrics(trajectories: Iterable[AgentTrajectory]) -> dict[str, Any]:
    return _compute(list(trajectories), difficulty=True)
