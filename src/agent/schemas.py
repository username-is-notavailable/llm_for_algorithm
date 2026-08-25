from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from src.verifier.judge import CaseResult, JudgeResult, TestCase


class ActionType(StrEnum):
    EXECUTE_CODE = "execute_code"
    FINAL = "final"


class ActionParseStatus(StrEnum):
    EXPLICIT = "explicit"
    MISSING_ACTION_FALLBACK = "missing_action_fallback"
    INVALID_ACTION_FALLBACK = "invalid_action_fallback"


class TerminationReason(StrEnum):
    SUCCESS = "success"
    FINAL_INCORRECT = "final_incorrect"
    EXECUTION_BUDGET_EXHAUSTED_AUTO_FINAL = "execution_budget_exhausted_auto_final"
    TOKEN_BUDGET_EXHAUSTED_AUTO_FINAL = "token_budget_exhausted_auto_final"
    REPEATED_CODE = "repeated_code"
    CODE_EXTRACTION_FAILED = "code_extraction_failed"
    MODEL_STOP_WITHOUT_CODE = "model_stop_without_code"
    SANDBOX_ERROR = "sandbox_error"


@dataclass(frozen=True)
class ExecutionLimits:
    compile_timeout_seconds: float = 10.0
    execution_timeout_seconds: float = 2.0
    memory_limit_bytes: int = 512 * 1024 * 1024
    output_limit_bytes: int = 1024 * 1024


@dataclass(frozen=True)
class AgentProblem:
    problem_id: str
    problem: str
    visible_tests: tuple[TestCase, ...]
    hidden_tests: tuple[TestCase, ...]
    difficulty: str | None = None
    source: str | None = None
    language: str = "cpp"
    environment_id: str = "cpp17-v1"
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.problem_id or not self.problem:
            raise ValueError("AgentProblem requires a problem_id and problem statement")
        if self.language != "cpp":
            raise ValueError("Code Agent v1 only supports cpp")
        if not self.visible_tests or not self.hidden_tests:
            raise ValueError("AgentProblem requires non-empty visible and hidden tests")


@dataclass(frozen=True)
class AgentConfig:
    protocol_version: str = "code-agent-v1"
    feedback_version: str = "execution-feedback-v1"
    max_execute_calls: int = 3
    max_candidate_submissions: int = 4
    max_total_generation_tokens: int = 32768
    max_feedback_bytes: int = 4096
    max_visible_failure_cases: int = 1
    stop_on_repeated_code: bool = True
    evaluate_hidden_each_submission: bool = True

    def __post_init__(self) -> None:
        if self.max_execute_calls < 0:
            raise ValueError("max_execute_calls must be non-negative")
        if self.max_candidate_submissions < 1:
            raise ValueError("max_candidate_submissions must be positive")
        if self.max_candidate_submissions < self.max_execute_calls + 1:
            raise ValueError("max_candidate_submissions must leave room for a final submission")
        if self.max_total_generation_tokens < 1:
            raise ValueError("max_total_generation_tokens must be positive")
        if self.max_feedback_bytes < 1 or self.max_visible_failure_cases != 1:
            raise ValueError("Code Agent v1 requires positive feedback bytes and one failure case")


@dataclass(frozen=True)
class CandidateSubmission:
    turn: int
    response: str
    code: str | None
    code_sha256: str | None
    requested_action: str | None
    effective_action: ActionType
    action_parse_status: ActionParseStatus
    prompt_tokens: int | None
    generation_tokens: int
    finish_reason: str | None
    reasoning_content: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionObservation:
    status: str
    judge: JudgeResult
    model_feedback: str
    executions_remaining: int
    cached: bool = False

    @property
    def visible_pass_rate(self) -> float:
        return self.judge.pass_rate


@dataclass(frozen=True)
class HiddenEvaluation:
    judge: JudgeResult

    @property
    def success(self) -> bool:
        return self.judge.passed == self.judge.total


@dataclass(frozen=True)
class AgentStep:
    turn: int
    prompt_messages: list[dict[str, str]]
    submission: CandidateSubmission
    observation: ExecutionObservation | None
    hidden_evaluation: HiddenEvaluation | None
    previous_visible_pass_rate: float | None
    current_visible_pass_rate: float | None
    delta_visible_pass_rate: float | None


@dataclass
class AgentTrajectory:
    schema_version: str
    trajectory_id: str
    problem_id: str
    difficulty: str | None
    model: dict[str, Any]
    agent_config: AgentConfig
    hidden_tests_total: int
    steps: list[AgentStep] = field(default_factory=list)
    hidden_evaluation: HiddenEvaluation | None = None
    termination_reason: TerminationReason | None = None

    @property
    def execute_calls(self) -> int:
        return sum(step.observation is not None for step in self.steps)

    @property
    def candidate_submissions(self) -> int:
        return len(self.steps)

    @property
    def total_generation_tokens(self) -> int:
        return sum(step.submission.generation_tokens for step in self.steps)

    @property
    def first_attempt_success(self) -> bool:
        if not self.steps:
            return False
        return bool(self.steps[0].hidden_evaluation and self.steps[0].hidden_evaluation.success)

    @property
    def final_success(self) -> bool:
        return bool(self.hidden_evaluation and self.hidden_evaluation.success)

    @property
    def repaired(self) -> bool:
        return not self.first_attempt_success and self.final_success

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outcome"] = {
            "first_attempt_success": self.first_attempt_success,
            "final_success": self.final_success,
            "repaired": self.repaired,
            "execute_calls": self.execute_calls,
            "candidate_submissions": self.candidate_submissions,
            "total_generation_tokens": self.total_generation_tokens,
            "termination_reason": self.termination_reason,
        }
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentTrajectory:
        def load_judge(data: dict[str, Any]) -> JudgeResult:
            copied = dict(data)
            copied["cases"] = [CaseResult(**case) for case in copied.get("cases", [])]
            return JudgeResult(**copied)

        def load_hidden(data: dict[str, Any] | None) -> HiddenEvaluation | None:
            return HiddenEvaluation(load_judge(data["judge"])) if data else None

        steps = []
        for raw in value.get("steps", []):
            submission_data = dict(raw["submission"])
            submission_data["effective_action"] = ActionType(submission_data["effective_action"])
            submission_data["action_parse_status"] = ActionParseStatus(
                submission_data["action_parse_status"]
            )
            observation_data = raw.get("observation")
            observation = None
            if observation_data:
                observation = ExecutionObservation(
                    status=observation_data["status"],
                    judge=load_judge(observation_data["judge"]),
                    model_feedback=observation_data["model_feedback"],
                    executions_remaining=int(observation_data["executions_remaining"]),
                    cached=bool(observation_data.get("cached", False)),
                )
            steps.append(
                AgentStep(
                    turn=int(raw["turn"]),
                    prompt_messages=list(raw.get("prompt_messages", [])),
                    submission=CandidateSubmission(**submission_data),
                    observation=observation,
                    hidden_evaluation=load_hidden(raw.get("hidden_evaluation")),
                    previous_visible_pass_rate=raw.get("previous_visible_pass_rate"),
                    current_visible_pass_rate=raw.get("current_visible_pass_rate"),
                    delta_visible_pass_rate=raw.get("delta_visible_pass_rate"),
                )
            )
        termination = value.get("termination_reason") or value.get("outcome", {}).get(
            "termination_reason"
        )
        return cls(
            schema_version=value["schema_version"],
            trajectory_id=value["trajectory_id"],
            problem_id=value["problem_id"],
            difficulty=value.get("difficulty"),
            model=dict(value.get("model", {})),
            agent_config=AgentConfig(**value["agent_config"]),
            hidden_tests_total=int(
                value.get("hidden_tests_total")
                or (value.get("hidden_evaluation") or {}).get("judge", {}).get("total", 0)
            ),
            steps=steps,
            hidden_evaluation=load_hidden(value.get("hidden_evaluation")),
            termination_reason=TerminationReason(termination) if termination else None,
        )
