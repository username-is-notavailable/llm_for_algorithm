from __future__ import annotations

import collections
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.data.repair_queue import RepairQueue
from src.inference.dashscope import DashScopeAgentGenerator, SlidingWindowLimiter
from src.verifier import TestCase, extract_code, judge


def distillation_prompt(problem: str, *, escalated: bool = False) -> str:
    escalation = (
        "A smaller teacher did not produce a fully verified solution. Solve the problem "
        "independently; do not discuss that earlier attempt.\n\n"
        if escalated
        else ""
    )
    return f"""{problem.strip()}

{escalation}Produce a concise training answer for a smaller code model. In the visible answer:
- give only the key algorithm, correctness idea, complexity, and essential edge cases;
- keep the explanation under 300 words and do not restate the problem;
- then emit exactly one complete GNU C++17 program in a ```cpp code fence.

Do not use tool/action tags. Do not expose private chain-of-thought or a long trial-and-error
derivation. Even if the problem is difficult, stop analyzing and provide the best complete
program within the output budget."""


def has_obvious_repetition(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    return any(count >= 5 for count in collections.Counter(lines).values())


def _judge_summary(result: Any) -> dict[str, Any]:
    value = asdict(result)
    value.pop("cases", None)
    return value


def process_distillation_task(
    payload: dict[str, Any],
    config: dict[str, Any],
    teacher: dict[str, Any],
    generator: DashScopeAgentGenerator,
    *,
    escalated: bool,
) -> tuple[dict[str, Any], bool]:
    generated = generator.generate(
        [{"role": "user", "content": distillation_prompt(payload["problem"], escalated=escalated)}],
        config["generation"],
    )
    response = generated.text.strip()
    code = extract_code(response)
    visible_tokens_estimated = max(1, len(response) // 4) if response else 0
    maximum_visible_tokens = int(config["acceptance"].get("max_visible_tokens_estimated", 4096))
    repetition = has_obvious_repetition(response)
    result: dict[str, Any] = {
        "schema_version": "verified-distillation-candidate-v1",
        "task_id": f"{teacher['stage']}:{payload['problem_id']}",
        "problem_id": payload["problem_id"],
        "difficulty": payload.get("difficulty"),
        "source": payload.get("source"),
        "teacher_stage": teacher["stage"],
        "teacher_model": teacher["model"],
        "escalated": escalated,
        "problem": payload,
        "response": response,
        "code": code,
        "reasoning_content": generated.reasoning_content,
        "finish_reason": generated.finish_reason,
        "completion_tokens": generated.token_count,
        "visible_tokens_estimated": visible_tokens_estimated,
        "provider_metadata": generated.provider_metadata,
        "obvious_repetition": repetition,
    }
    if not code:
        return {**result, "accepted": False, "rejection_reason": "no_code"}, False

    tests = [TestCase(**test) for test in payload["visible_tests"] + payload["hidden_tests"]]
    verifier = config["verifier"]
    judged = judge(
        code,
        tests,
        compile_timeout_seconds=float(verifier["compile_timeout_seconds"]),
        execution_timeout_seconds=float(verifier["execution_timeout_seconds"]),
        memory_limit_bytes=int(verifier["memory_limit_mb"]) * 1024 * 1024,
        output_limit_bytes=int(verifier["output_limit_bytes"]),
    )
    result["judge"] = _judge_summary(judged)
    rejection = None
    if judged.passed != judged.total:
        rejection = "verification_failed"
    elif generated.finish_reason != "stop":
        rejection = "incomplete_generation"
    elif visible_tokens_estimated > maximum_visible_tokens:
        rejection = "visible_response_too_long"
    elif repetition:
        rejection = "obvious_repetition"
    accepted = rejection is None
    return {**result, "accepted": accepted, "rejection_reason": rejection}, accepted


def run_distillation_workers(
    queue: RepairQueue,
    config: dict[str, Any],
    teacher: dict[str, Any],
    *,
    escalated: bool,
    generator_factory: Callable[[], DashScopeAgentGenerator] | None = None,
) -> dict[str, int]:
    api = {**config["api"], **teacher}
    limiter = SlidingWindowLimiter(
        requests_per_minute=int(api["requests_per_minute"]),
        tokens_per_minute=int(api["tokens_per_minute"]),
    )
    if generator_factory is None:
        generator_factory = lambda: DashScopeAgentGenerator(api, limiter=limiter)
    max_attempts = int(config["queue"].get("max_task_attempts", 1))
    progress_interval = float(config["queue"].get("progress_interval_seconds", 10))
    progress_stop = threading.Event()

    def report_progress() -> None:
        while not progress_stop.wait(progress_interval):
            counts = queue.counts()
            total = sum(counts.values())
            finished = sum(counts.get(key, 0) for key in ("accepted", "rejected", "failed"))
            print(
                f"{teacher['stage']} progress: {finished}/{total} finished | "
                f"accepted={counts.get('accepted', 0)} rejected={counts.get('rejected', 0)} "
                f"failed={counts.get('failed', 0)} running={counts.get('running', 0)} "
                f"pending={counts.get('pending', 0)}",
                flush=True,
            )

    def worker(index: int) -> None:
        generator = generator_factory()
        worker_id = f"{teacher['stage']}-worker-{index:03d}-{threading.get_ident()}"
        while True:
            claimed = queue.claim(worker_id)
            if claimed is None:
                return
            task_id, payload, attempts = claimed
            try:
                result, accepted = process_distillation_task(
                    payload, config, teacher, generator, escalated=escalated
                )
                queue.complete(task_id, result, accepted=accepted)
            except Exception as error:
                queue.fail(
                    task_id,
                    f"{type(error).__name__}: {error}",
                    retry=attempts < max_attempts,
                )

    progress = threading.Thread(target=report_progress, name=f"{teacher['stage']}-progress", daemon=True)
    progress.start()
    try:
        with ThreadPoolExecutor(max_workers=int(api["concurrency"])) as executor:
            futures = [executor.submit(worker, index) for index in range(int(api["concurrency"]))]
            for future in futures:
                future.result()
    finally:
        progress_stop.set()
        progress.join()
    return queue.counts()


def export_queue(queue: RepairQueue, output_dir: Path, prefix: str) -> None:
    import json

    for status in ("accepted", "rejected"):
        with (output_dir / f"{prefix}_{status}.jsonl").open("w", encoding="utf-8") as handle:
            for row in queue.export(status):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
