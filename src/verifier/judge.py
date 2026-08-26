from __future__ import annotations

import tempfile
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
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
    actual_lines = [line.strip() for line in actual.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if len(actual_lines) != len(expected_lines):
        return False
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line == expected_line:
            continue
        try:
            actual_numbers = [Decimal(token) for token in actual_line.split()]
            expected_numbers = [Decimal(token) for token in expected_line.split()]
        except InvalidOperation:
            return False
        if actual_numbers != expected_numbers:
            return False
    return True


def _compile_output_checker(source: str, workdir: Path, testlib_path: str | Path | None, timeout: float) -> tuple[Path | None, str]:
    source_path = workdir / "checker.cpp"
    binary_path = workdir / "checker"
    source_path.write_text(source, encoding="utf-8")
    if testlib_path is not None:
        header = Path(testlib_path)
        if not header.is_file():
            raise FileNotFoundError(f"Missing checker dependency: {header}")
        shutil.copy2(header, workdir / "testlib.h")
    try:
        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", str(source_path), "-o", str(binary_path)],
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "output checker compilation timed out"
    stderr = result.stderr[: 64 * 1024].decode("utf-8", errors="replace")
    return (binary_path if result.returncode == 0 else None), stderr


def _check_output(
    checker: Path, test_case: TestCase, actual: str, workdir: Path, index: int, timeout: float
) -> tuple[bool, str]:
    input_path = workdir / f"checker-{index}.in"
    actual_path = workdir / f"checker-{index}.out"
    answer_path = workdir / f"checker-{index}.ans"
    input_path.write_text(test_case.input, encoding="utf-8")
    actual_path.write_text(actual, encoding="utf-8")
    answer_path.write_text(test_case.output, encoding="utf-8")
    try:
        result = subprocess.run(
            [str(checker), str(input_path), str(actual_path), str(answer_path)],
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "output checker timed out"
    return result.returncode == 0, result.stderr[: 64 * 1024].decode("utf-8", errors="replace")


def judge(
    code: str,
    test_cases: Iterable[TestCase | Mapping[str, str]],
    *,
    compile_timeout_seconds: float = 10.0,
    execution_timeout_seconds: float = 2.0,
    memory_limit_bytes: int = 512 * 1024 * 1024,
    output_limit_bytes: int = 1024 * 1024,
    output_checker_source: str | None = None,
    testlib_path: str | Path | None = None,
    checker_timeout_seconds: float = 4.0,
) -> JudgeResult:
    cases = [_coerce_test_case(test_case) for test_case in test_cases]
    if not cases:
        raise ValueError("At least one test case is required")

    with tempfile.TemporaryDirectory(prefix="qwen3-verifier-") as temporary_directory:
        workdir = Path(temporary_directory)
        checker = None
        if output_checker_source:
            checker, checker_stderr = _compile_output_checker(
                output_checker_source, workdir, testlib_path, compile_timeout_seconds
            )
            if checker is None:
                return JudgeResult(
                    compiled=False,
                    passed=0,
                    total=len(cases),
                    pass_rate=0.0,
                    runtime_error=False,
                    timeout="timed out" in checker_stderr,
                    error_type="checker_compile_error",
                    compile_stderr=checker_stderr,
                )
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
        for case_index, test_case in enumerate(cases):
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
            elif checker is not None:
                accepted, checker_stderr = _check_output(
                    checker,
                    test_case,
                    execution.stdout,
                    workdir,
                    case_index,
                    checker_timeout_seconds,
                )
                error_type = None if accepted else "wrong_answer"
            else:
                error_type = None if _outputs_match(execution.stdout, test_case.output) else "wrong_answer"
            case_results.append(
                CaseResult(
                    passed=error_type is None,
                    error_type=error_type,
                    expected=test_case.output,
                    actual=execution.stdout,
                    stderr=execution.stderr or (checker_stderr if checker is not None else ""),
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
