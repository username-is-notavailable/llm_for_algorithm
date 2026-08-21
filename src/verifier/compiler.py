from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileResult:
    success: bool
    binary_path: Path | None
    stderr: str
    timed_out: bool
    return_code: int | None


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def compile_code(
    code: str,
    workdir: Path,
    *,
    timeout_seconds: float = 10.0,
    compiler: str = "g++",
    max_stderr_bytes: int = 64 * 1024,
) -> CompileResult:
    source_path = workdir / "source.cpp"
    binary_path = workdir / "program"
    source_path.write_text(code, encoding="utf-8")
    command = [compiler, "-std=c++17", "-O2", "-pipe", str(source_path), "-o", str(binary_path)]
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr_bytes = process.communicate(timeout=timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        _, stderr_bytes = process.communicate()
        timed_out = True

    stderr = stderr_bytes[:max_stderr_bytes].decode("utf-8", errors="replace")
    success = not timed_out and process.returncode == 0 and binary_path.is_file()
    return CompileResult(
        success=success,
        binary_path=binary_path if success else None,
        stderr=stderr,
        timed_out=timed_out,
        return_code=process.returncode,
    )
