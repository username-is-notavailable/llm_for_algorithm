from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from src.verifier.compiler import compile_code
from src.verifier.executor import execute_binary


@dataclass(frozen=True)
class TestCase:
    input: str
    output: str


@dataclass(frozen=True)
class CaseResult:
    passed: bool
    error_type: str | None
    expected: str
    actual: str
    stderr: str
    return_code: int | None
    duration_seconds: float


@dataclass(frozen=True)
class JudgeResult:
    compiled: bool
    passed: int
    total: int
    pass_rate: float
    runtime_error: bool
    timeout: bool
    error_type: str | None
    compile_stderr: str = ""
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_test_case(value: TestCase | Mapping[str, str]) -> TestCase:
    if isinstance(value, TestCase):
        return value
    try:
        stdin = value["input"]
        expected = value["output"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Each test case must contain string input and output fields") from exc
    if not isinstance(stdin, str) or not isinstance(expected, str):
        raise ValueError("Each test case must contain string input and output fields")
    return TestCase(input=stdin, output=expected)


def _outputs_match(actual: str, expected: str) -> bool:
    return actual.split() == expected.split()


def judge(
    code: str,
    test_cases: Iterable[TestCase | Mapping[str, str]],
    *,
    compile_timeout_seconds: float = 10.0,
    execution_timeout_seconds: float = 2.0,
    memory_limit_bytes: int = 512 * 1024 * 1024,
    output_limit_bytes: int = 1024 * 1024,
) -> JudgeResult:
    cases = [_coerce_test_case(test_case) for test_case in test_cases]
    if not cases:
        raise ValueError("At least one test case is required")

    with tempfile.TemporaryDirectory(prefix="qwen3-verifier-") as temporary_directory:
        workdir = Path(temporary_directory)
        compilation = compile_code(code, workdir, timeout_seconds=compile_timeout_seconds)
        if not compilation.success:
            error_type = "compile_timeout" if compilation.timed_out else "compile_error"
            return JudgeResult(
                compiled=False,
                passed=0,
                total=len(cases),
                pass_rate=0.0,
                runtime_error=False,
                timeout=compilation.timed_out,
                error_type=error_type,
                compile_stderr=compilation.stderr,
            )

        assert compilation.binary_path is not None
        case_results: list[CaseResult] = []
        for test_case in cases:
            execution = execute_binary(
                compilation.binary_path,
                test_case.input,
                workdir,
                timeout_seconds=execution_timeout_seconds,
                memory_limit_bytes=memory_limit_bytes,
                output_limit_bytes=output_limit_bytes,
            )
            if execution.timed_out:
                error_type = "timeout"
            elif execution.output_limit_exceeded:
                error_type = "output_limit"
            elif execution.runtime_error:
                error_type = "runtime_error"
            elif not _outputs_match(execution.stdout, test_case.output):
                error_type = "wrong_answer"
            else:
                error_type = None
            case_results.append(
                CaseResult(
                    passed=error_type is None,
                    error_type=error_type,
                    expected=test_case.output,
                    actual=execution.stdout,
                    stderr=execution.stderr,
                    return_code=execution.return_code,
                    duration_seconds=execution.duration_seconds,
                )
            )

    passed = sum(case.passed for case in case_results)
    first_error = next((case.error_type for case in case_results if case.error_type), None)
    return JudgeResult(
        compiled=True,
        passed=passed,
        total=len(cases),
        pass_rate=passed / len(cases),
        runtime_error=any(case.error_type == "runtime_error" for case in case_results),
        timeout=any(case.error_type == "timeout" for case in case_results),
        error_type=first_error,
        cases=case_results,
    )
