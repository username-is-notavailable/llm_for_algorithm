from __future__ import annotations

from src.verifier.judge import JudgeResult


def observation_status(result: JudgeResult) -> str:
    if not result.compiled:
        return "compile_timeout" if result.error_type == "compile_timeout" else "compile_error"
    if result.passed == result.total:
        return "passed_visible_tests"
    return str(result.error_type or "wrong_answer")


def _truncate(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = f"\n[truncated after {max_bytes} bytes]"
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        return suffix_bytes[:max_bytes].decode("utf-8", errors="ignore")
    budget = max_bytes - len(suffix_bytes)
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def format_execution_feedback(
    result: JudgeResult,
    *,
    executions_remaining: int,
    max_bytes: int,
    first_failing_input: str | None = None,
) -> str:
    status = observation_status(result)
    lines = [
        "Execution result:",
        f"- Status: {status.upper()}",
        f"- Visible tests passed: {result.passed}/{result.total}",
    ]
    if not result.compiled:
        lines.extend(["- Compiler output:", result.compile_stderr or "(empty)"])
    else:
        first_failure = next((case for case in result.cases if not case.passed), None)
        if first_failure is not None:
            lines.extend(
                [
                    f"- Failure type: {str(first_failure.error_type).upper()}",
                    "- First failing input:",
                    first_failing_input or "(unavailable)",
                ]
            )
            # JudgeResult deliberately stores only expected/actual. The input is
            # injected by the backend below when it still has the visible cases.
            lines.extend(
                [
                    "- Expected output:",
                    first_failure.expected,
                    "- Actual output:",
                    first_failure.actual,
                ]
            )
            if first_failure.stderr:
                lines.extend(["- Stderr:", first_failure.stderr])
    lines.append(f"- Executions remaining: {executions_remaining}")
    return _truncate("\n".join(lines), max_bytes)
