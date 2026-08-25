from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.agent.backend import LocalVerifierBackend
from src.agent.controller import run_agent
from src.agent.schemas import (
    ActionParseStatus,
    ActionType,
    AgentConfig,
    AgentProblem,
    ExecutionLimits,
)
from src.data.repair_queue import RepairQueue
from src.inference.dashscope import DashScopeAgentGenerator, SlidingWindowLimiter
from src.verifier import TestCase


def load_problem(row: dict[str, Any], verifier: dict[str, Any]) -> AgentProblem:
    return AgentProblem(
        problem_id=row["problem_id"],
        problem=row["problem"],
        visible_tests=tuple(TestCase(**test) for test in row["visible_tests"]),
        hidden_tests=tuple(TestCase(**test) for test in row["hidden_tests"]),
        difficulty=row.get("difficulty"),
        source=row.get("source"),
        limits=ExecutionLimits(
            compile_timeout_seconds=float(verifier["compile_timeout_seconds"]),
            execution_timeout_seconds=float(verifier["execution_timeout_seconds"]),
            memory_limit_bytes=int(verifier["memory_limit_mb"]) * 1024 * 1024,
            output_limit_bytes=int(verifier["output_limit_bytes"]),
        ),
        metadata=row.get("metadata", {}),
    )


def repair_prompt(problem: str, code: str, feedback: str) -> str:
    return f"""{problem.strip()}

An earlier student submitted the following GNU C++17 program:
```cpp
{code.strip()}
```

It received this real execution-environment observation:
{feedback.strip()}

Analyze the concrete failure, repair the program, and follow the action protocol. Do not invent
test results. Reason briefly and directly; do not restate the problem or repeat the full derivation.
In the visible answer, identify the specific cause in at most five short bullet points, then make
the smallest necessary correction and emit exactly one complete program. Use execute_code when another real
execution is needed. After a passed observation, do not redesign or regenerate a different
solution: immediately use final and return the last passing complete program."""


def process_task(
    payload: dict[str, Any],
    config: dict[str, Any],
    generator: DashScopeAgentGenerator,
) -> tuple[dict[str, Any], bool]:
    backend = LocalVerifierBackend()
    original = load_problem(payload["problem"], config["verifier"])
    initial = payload["initial_submission"]
    initial_code = initial["code"]
    observation = backend.execute_visible(
        initial_code,
        original,
        executions_remaining=int(config["agent"]["max_execute_calls"]),
        max_feedback_bytes=int(config["agent"].get("max_feedback_bytes", 4096)),
    )
    initial_hidden = backend.evaluate_hidden(initial_code, original)
    base = {
        "schema_version": "repair-example-v1",
        "task_id": payload["task_id"],
        "problem_id": original.problem_id,
        "difficulty": original.difficulty,
        "source": original.source,
        "failure_producer_model": initial["producer_model"],
        "teacher_model": config["api"]["model"],
        "initial_submission": initial,
        "initial_observation": asdict(observation),
        "initial_hidden_evaluation": asdict(initial_hidden),
    }
    if initial_hidden.success:
        return {**base, "accepted": False, "rejection_reason": "initial_code_already_correct"}, False
    if observation.visible_pass_rate == 1:
        return {**base, "accepted": False, "rejection_reason": "no_visible_failure_feedback"}, False
    repair_problem = AgentProblem(
        problem_id=original.problem_id,
        problem=repair_prompt(original.problem, initial_code, observation.model_feedback),
        visible_tests=original.visible_tests,
        hidden_tests=original.hidden_tests,
        difficulty=original.difficulty,
        source=original.source,
        limits=original.limits,
        metadata=original.metadata,
    )
    trajectory = run_agent(
        trajectory_id=f"m10-api-repair:{payload['task_id']}",
        problem=repair_problem,
        model={"provider": "dashscope", "name_or_path": config["api"]["model"]},
        config=AgentConfig(**config["agent"]),
        generator=generator,
        backend=backend,
        generation=config["generation"],
    )
    explicit = all(
        step.submission.action_parse_status == ActionParseStatus.EXPLICIT for step in trajectory.steps
    )
    final_explicit = bool(
        trajectory.steps
        and trajectory.steps[-1].submission.requested_action == ActionType.FINAL.value
    )
    accepted = trajectory.final_success and explicit and final_explicit
    rejection = None
    if not trajectory.final_success:
        rejection = "repair_failed_full_tests"
    elif not explicit:
        rejection = "invalid_action_protocol"
    elif not final_explicit:
        rejection = "missing_explicit_final"
    return {
        **base,
        "accepted": accepted,
        "rejection_reason": rejection,
        "repair_trajectory": trajectory.to_dict(),
    }, accepted


def run_workers(
    queue: RepairQueue,
    config: dict[str, Any],
    *,
    generator_factory: Callable[[], DashScopeAgentGenerator] | None = None,
) -> dict[str, int]:
    api = config["api"]
    limiter = SlidingWindowLimiter(
        requests_per_minute=int(api["requests_per_minute"]),
        tokens_per_minute=int(api["tokens_per_minute"]),
    )
    if generator_factory is None:
        generator_factory = lambda: DashScopeAgentGenerator(api, limiter=limiter)
    max_task_attempts = int(config["queue"].get("max_task_attempts", 1))

    def worker(index: int) -> None:
        generator = generator_factory()
        worker_id = f"api-worker-{index:03d}-{threading.get_ident()}"
        while True:
            target = config["queue"].get("target_accepted")
            if target is not None and queue.counts().get("accepted", 0) >= int(target):
                return
            claimed = queue.claim(worker_id)
            if claimed is None:
                return
            task_id, payload, attempts = claimed
            try:
                result, accepted = process_task(payload, config, generator)
                queue.complete(task_id, result, accepted=accepted)
            except Exception as error:
                # API-level retries happen inside the generator. A task-level retry
                # reruns the full trajectory and is intentionally conservative.
                queue.fail(task_id, f"{type(error).__name__}: {error}", retry=attempts < max_task_attempts)

    with ThreadPoolExecutor(max_workers=int(api["concurrency"])) as executor:
        futures = [executor.submit(worker, index) for index in range(int(api["concurrency"]))]
        for future in futures:
            future.result()
    return queue.counts()


def export_results(queue: RepairQueue, output_dir: Path) -> None:
    for status, filename in (("accepted", "accepted.jsonl"), ("rejected", "rejected.jsonl")):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            for row in queue.export(status):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
