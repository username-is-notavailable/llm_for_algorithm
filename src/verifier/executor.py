from __future__ import annotations

import math
import os
import resource
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool
    output_limit_exceeded: bool
    duration_seconds: float

    @property
    def runtime_error(self) -> bool:
        return not self.timed_out and self.return_code not in (0, None)


def _set_resource_limits(timeout_seconds: float, memory_limit_bytes: int, output_limit_bytes: int) -> None:
    cpu_soft = max(1, math.ceil(timeout_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit_bytes, output_limit_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_limited(path: Path, limit: int) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        content = handle.read(limit)
    return content.decode("utf-8", errors="replace"), size >= limit


def execute_binary(
    binary_path: Path,
    stdin: str,
    workdir: Path,
    *,
    timeout_seconds: float = 2.0,
    memory_limit_bytes: int = 512 * 1024 * 1024,
    output_limit_bytes: int = 1024 * 1024,
) -> ExecutionResult:
    stdout_path = workdir / "stdout.txt"
    stderr_path = workdir / "stderr.txt"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            [str(binary_path.resolve())],
            cwd=workdir,
            stdin=subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            preexec_fn=lambda: _set_resource_limits(
                timeout_seconds, memory_limit_bytes, output_limit_bytes
            ),
        )
        try:
            process.communicate(input=stdin.encode("utf-8"), timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            process.communicate()
            timed_out = True

    duration = time.monotonic() - started
    stdout, stdout_limited = _read_limited(stdout_path, output_limit_bytes)
    stderr, stderr_limited = _read_limited(stderr_path, output_limit_bytes)
    output_limited = stdout_limited or stderr_limited or process.returncode == -signal.SIGXFSZ
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        return_code=process.returncode,
        timed_out=timed_out,
        output_limit_exceeded=output_limited,
        duration_seconds=duration,
    )
