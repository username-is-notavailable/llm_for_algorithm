from __future__ import annotations

import hashlib
from typing import Any, Protocol

from src.agent.backend import ExecutionBackend, full_gate_problem, reveal_hidden_counterexample
from src.agent.schemas import (
    ActionType,
    AgentConfig,
    AgentProblem,
    AgentStep,
    AgentTrajectory,
    CandidateSubmission,
    TerminationReason,
)
from src.agent.protocol import parse_submission
from src.inference.generate import GeneratedText


class AgentGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText: ...


def build_initial_messages(problem: AgentProblem, config: AgentConfig) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a competitive programming agent. Produce complete GNU C++17 programs. "
                "Choose <action>execute_code</action> to run visible tests and receive feedback, "
                "or <action>final</action> to submit your final program. "
                f"You may request execution feedback at most {config.max_execute_calls} times. "
                "After executing a program, <action>final</action> by itself submits the most "
                "recently executed program; do not repeat that program. If no program has been "
                "executed yet, a final action must include one complete program. "
                "If current feedback tests pass but private validation fails, the environment may "
                "reveal one counterexample; treat it as a new feedback test. "
                "Never omit the action tag. An execute response must contain exactly one complete "
                "program in a ```cpp code fence."
            ),
        },
        {"role": "user", "content": problem.problem.strip()},
    ]


def _termination_for_hidden(success: bool, *, auto_final: bool, token_final: bool) -> TerminationReason:
    if token_final:
        return TerminationReason.TOKEN_BUDGET_EXHAUSTED_AUTO_FINAL
    if auto_final:
        return TerminationReason.EXECUTION_BUDGET_EXHAUSTED_AUTO_FINAL
    return TerminationReason.SUCCESS if success else TerminationReason.FINAL_INCORRECT


def run_agent(
    *,
    trajectory_id: str,
    problem: AgentProblem,
    model: dict[str, Any],
    config: AgentConfig,
    generator: AgentGenerator,
    backend: ExecutionBackend,
    generation: dict[str, Any] | None = None,
    initial_revealed_counterexamples: int = 0,
) -> AgentTrajectory:
    trajectory = AgentTrajectory(
        schema_version="agent-trajectory-v2",
        trajectory_id=trajectory_id,
        problem_id=problem.problem_id,
        difficulty=problem.difficulty,
        model=dict(model),
        agent_config=config,
        hidden_tests_total=len(problem.hidden_tests),
        full_tests_total=len(problem.visible_tests) + len(problem.hidden_tests),
        initial_revealed_counterexamples=initial_revealed_counterexamples,
    )
    messages = build_initial_messages(problem, config)
    generation_options = dict(generation or {})
    code_hashes: set[str] = set()
    last_executed_code: str | None = None
    last_visible_pass_rate: float | None = None
    active_problem = problem
    revealed_counterexamples = initial_revealed_counterexamples

    for turn in range(config.max_candidate_submissions):
        generated = generator.generate(messages, generation_options)
        parsed = parse_submission(
            generated.text,
            execute_calls=trajectory.execute_calls,
            max_execute_calls=config.max_execute_calls,
        )
        candidate_code = parsed.code
        reused_last_code = False
        if (
            candidate_code is None
            and parsed.action == ActionType.FINAL
            and last_executed_code is not None
        ):
            candidate_code = last_executed_code
            reused_last_code = True
        if candidate_code is None:
            trajectory.steps.append(
                AgentStep(
                    turn=turn,
                    prompt_messages=[dict(message) for message in messages],
                    submission=CandidateSubmission(
                        turn=turn,
                        response=generated.text,
                        code=None,
                        code_sha256=None,
                        requested_action=parsed.requested_action,
                        effective_action=parsed.action,
                        action_parse_status=parsed.parse_status,
                        prompt_tokens=None,
                        generation_tokens=generated.token_count,
                        finish_reason=generated.finish_reason,
                        reasoning_content=generated.reasoning_content,
                        provider_metadata=dict(generated.provider_metadata),
                    ),
                    observation=None,
                    hidden_evaluation=None,
                    previous_visible_pass_rate=last_visible_pass_rate,
                    current_visible_pass_rate=None,
                    delta_visible_pass_rate=None,
                )
            )
            trajectory.termination_reason = (
                TerminationReason.MODEL_STOP_WITHOUT_CODE
                if generated.finish_reason == "stop"
                else TerminationReason.CODE_EXTRACTION_FAILED
            )
            break

        code_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        token_total_after = trajectory.total_generation_tokens + generated.token_count
        token_auto_final = token_total_after >= config.max_total_generation_tokens
        execution_auto_final = (
            parsed.action == ActionType.EXECUTE_CODE
            and trajectory.execute_calls >= config.max_execute_calls
        )
        final_slot = turn == config.max_candidate_submissions - 1
        slot_auto_final = final_slot and parsed.action != ActionType.FINAL
        effective_action = (
            ActionType.FINAL
            if parsed.action == ActionType.FINAL
            or execution_auto_final
            or token_auto_final
            or slot_auto_final
            else ActionType.EXECUTE_CODE
        )
        provider_metadata = dict(generated.provider_metadata)
        if reused_last_code:
            provider_metadata["final_reused_last_code"] = True
        submission = CandidateSubmission(
            turn=turn,
            response=generated.text,
            code=candidate_code,
            code_sha256=code_hash,
            requested_action=parsed.requested_action,
            effective_action=effective_action,
            action_parse_status=parsed.parse_status,
            prompt_tokens=None,
            generation_tokens=generated.token_count,
            finish_reason=generated.finish_reason,
            reasoning_content=generated.reasoning_content,
            provider_metadata=provider_metadata,
        )

        if (
            config.stop_on_repeated_code
            and effective_action == ActionType.EXECUTE_CODE
            and code_hash in code_hashes
        ):
            trajectory.steps.append(
                AgentStep(
                    turn=turn,
                    prompt_messages=[dict(message) for message in messages],
                    submission=submission,
                    observation=None,
                    hidden_evaluation=None,
                    previous_visible_pass_rate=last_visible_pass_rate,
                    current_visible_pass_rate=None,
                    delta_visible_pass_rate=None,
                )
            )
            try:
                trajectory.hidden_evaluation = backend.evaluate_hidden(
                    candidate_code, full_gate_problem(active_problem)
                )
            except Exception:
                trajectory.termination_reason = TerminationReason.SANDBOX_ERROR
                break
            trajectory.steps[-1] = AgentStep(
                turn=trajectory.steps[-1].turn,
                prompt_messages=trajectory.steps[-1].prompt_messages,
                submission=trajectory.steps[-1].submission,
                observation=None,
                hidden_evaluation=trajectory.hidden_evaluation,
                previous_visible_pass_rate=trajectory.steps[-1].previous_visible_pass_rate,
                current_visible_pass_rate=None,
                delta_visible_pass_rate=None,
            )
            trajectory.termination_reason = TerminationReason.REPEATED_CODE
            break
        code_hashes.add(code_hash)

        if effective_action == ActionType.EXECUTE_CODE:
            last_executed_code = candidate_code

        if effective_action == ActionType.FINAL:
            trajectory.steps.append(
                AgentStep(
                    turn=turn,
                    prompt_messages=[dict(message) for message in messages],
                    submission=submission,
                    observation=None,
                    hidden_evaluation=None,
                    previous_visible_pass_rate=last_visible_pass_rate,
                    current_visible_pass_rate=None,
                    delta_visible_pass_rate=None,
                )
            )
            try:
                trajectory.hidden_evaluation = backend.evaluate_hidden(
                    candidate_code, full_gate_problem(active_problem)
                )
            except Exception:
                trajectory.termination_reason = TerminationReason.SANDBOX_ERROR
                break
            trajectory.steps[-1] = AgentStep(
                turn=trajectory.steps[-1].turn,
                prompt_messages=trajectory.steps[-1].prompt_messages,
                submission=trajectory.steps[-1].submission,
                observation=None,
                hidden_evaluation=trajectory.hidden_evaluation,
                previous_visible_pass_rate=trajectory.steps[-1].previous_visible_pass_rate,
                current_visible_pass_rate=None,
                delta_visible_pass_rate=None,
            )
            trajectory.termination_reason = _termination_for_hidden(
                trajectory.hidden_evaluation.success,
                auto_final=execution_auto_final or slot_auto_final,
                token_final=token_auto_final,
            )
            break

        executions_remaining = config.max_execute_calls - trajectory.execute_calls - 1
        try:
            observation = backend.execute_visible(
                candidate_code,
                active_problem,
                executions_remaining=executions_remaining,
                max_feedback_bytes=config.max_feedback_bytes,
            )
        except Exception:
            trajectory.steps.append(
                AgentStep(
                    turn=turn,
                    prompt_messages=[dict(message) for message in messages],
                    submission=submission,
                    observation=None,
                    hidden_evaluation=None,
                    previous_visible_pass_rate=last_visible_pass_rate,
                    current_visible_pass_rate=None,
                    delta_visible_pass_rate=None,
                )
            )
            trajectory.termination_reason = TerminationReason.SANDBOX_ERROR
            break
        current_pass_rate = observation.visible_pass_rate
        hidden_evaluation = None
        if config.evaluate_hidden_each_submission:
            try:
                hidden_evaluation = backend.evaluate_hidden(candidate_code, active_problem)
            except Exception:
                trajectory.termination_reason = TerminationReason.SANDBOX_ERROR
        if (
            trajectory.termination_reason is None
            and observation.visible_pass_rate == 1
            and hidden_evaluation is not None
            and not hidden_evaluation.success
            and revealed_counterexamples < config.max_revealed_counterexamples
        ):
            try:
                revealed = reveal_hidden_counterexample(
                    backend,
                    candidate_code,
                    active_problem,
                    hidden_evaluation,
                    executions_remaining=executions_remaining,
                    max_feedback_bytes=config.max_feedback_bytes,
                    min_private_tests=config.min_private_tests,
                )
            except Exception:
                trajectory.termination_reason = TerminationReason.SANDBOX_ERROR
                revealed = None
            if revealed is not None:
                active_problem, observation = revealed
                revealed_counterexamples += 1
                current_pass_rate = observation.visible_pass_rate
        trajectory.steps.append(
            AgentStep(
                turn=turn,
                prompt_messages=[dict(message) for message in messages],
                submission=submission,
                observation=observation,
                hidden_evaluation=hidden_evaluation,
                previous_visible_pass_rate=last_visible_pass_rate,
                current_visible_pass_rate=current_pass_rate,
                delta_visible_pass_rate=(
                    None if last_visible_pass_rate is None else current_pass_rate - last_visible_pass_rate
                ),
            )
        )
        if trajectory.termination_reason == TerminationReason.SANDBOX_ERROR:
            break
        last_visible_pass_rate = current_pass_rate
        messages.extend(
            [
                {"role": "assistant", "content": generated.text},
                {
                    "role": "tool",
                    "content": (
                        observation.model_feedback
                        + "\n\nProtocol reminder: begin the next visible response with exactly "
                        "<action>execute_code</action> to test a correction, or "
                        "<action>final</action> by itself if the most recently executed program "
                        "is final; do not repeat that program."
                    ),
                },
            ]
        )

    if trajectory.termination_reason is None:
        raise RuntimeError("Agent loop exhausted without a termination reason")
    return trajectory
