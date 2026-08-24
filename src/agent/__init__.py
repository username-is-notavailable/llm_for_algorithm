"""Bounded execution-guided code agent."""

from src.agent.backend import ExecutionBackend, LocalVerifierBackend
from src.agent.controller import AgentGenerator, run_agent
from src.agent.metrics import compute_agent_metrics
from src.agent.protocol import ParsedSubmission, parse_submission
from src.agent.schemas import (
    ActionParseStatus,
    ActionType,
    AgentConfig,
    AgentProblem,
    AgentStep,
    AgentTrajectory,
    CandidateSubmission,
    ExecutionLimits,
    ExecutionObservation,
    HiddenEvaluation,
    TerminationReason,
)

__all__ = [
    "ActionParseStatus",
    "ActionType",
    "AgentConfig",
    "AgentGenerator",
    "AgentProblem",
    "AgentStep",
    "AgentTrajectory",
    "CandidateSubmission",
    "ExecutionBackend",
    "ExecutionLimits",
    "ExecutionObservation",
    "HiddenEvaluation",
    "LocalVerifierBackend",
    "ParsedSubmission",
    "TerminationReason",
    "compute_agent_metrics",
    "parse_submission",
    "run_agent",
]
