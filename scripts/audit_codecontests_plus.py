from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from fnmatch import fnmatch
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, HfFileSystem

from src.verifier.compiler import compile_code
from src.verifier.executor import execute_binary


def stable_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['source']}:{row['id']}".encode()).digest(),
    )[:size]


def cpp_submissions(value: Any, language: str = "C++") -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item["code"]
        for item in value
        if isinstance(item, dict)
        and item.get("language") == language
        and isinstance(item.get("code"), str)
        and item["code"].strip()
    ]


def prepare_contest_submission(code: str) -> str:
    # Online judges define this macro; many archived submissions guard local freopen/debug code with it.
    return "#ifndef ONLINE_JUDGE\n#define ONLINE_JUDGE 1\n#endif\n" + code


def ensure_testlib(path: Path, revision: str) -> Path:
    if path.is_file() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/MikeMirzayanov/testlib/{revision}/testlib.h"
    temporary = path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)
    return path


def compile_checker(source: str, testlib: Path, workdir: Path, timeout: float) -> tuple[Path | None, str]:
    (workdir / "checker.cpp").write_text(source, encoding="utf-8")
    shutil.copy2(testlib, workdir / "testlib.h")
    binary = workdir / "checker"
    try:
        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", "checker.cpp", "-o", str(binary)],
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "checker_compile_timeout"
    stderr = result.stderr.decode("utf-8", errors="replace")[-8192:]
    return (binary if result.returncode == 0 else None), stderr


def judge_submission(
    code: str,
    tests: list[dict[str, str]],
    checker: Path,
    workdir: Path,
    execution: dict[str, Any],
) -> dict[str, Any]:
    solution_dir = workdir / hashlib.sha256(code.encode()).hexdigest()[:12]
    solution_dir.mkdir()
    compilation = compile_code(
        prepare_contest_submission(code),
        solution_dir,
        timeout_seconds=float(execution["compile_timeout_seconds"]),
    )
    if not compilation.success:
        return {"compiled": False, "passed": 0, "total": len(tests), "error": "compile_error"}

    assert compilation.binary_path is not None
    passed = 0
    first_error = None
    for index, test in enumerate(tests):
        run = execute_binary(
            compilation.binary_path,
            test["input"],
            solution_dir,
            timeout_seconds=float(execution["execution_timeout_seconds"]),
            memory_limit_bytes=int(execution["memory_limit_bytes"]),
            output_limit_bytes=int(execution["output_limit_bytes"]),
        )
        if run.timed_out:
            error = "timeout"
        elif run.output_limit_exceeded:
            error = "output_limit"
        elif run.runtime_error:
            error = "runtime_error"
        else:
            input_path = solution_dir / f"case-{index}.in"
            actual_path = solution_dir / f"case-{index}.out"
            answer_path = solution_dir / f"case-{index}.ans"
            input_path.write_text(test["input"], encoding="utf-8")
            actual_path.write_text(run.stdout, encoding="utf-8")
            answer_path.write_text(test["output"], encoding="utf-8")
            try:
                checked = subprocess.run(
                    [str(checker), str(input_path), str(actual_path), str(answer_path)],
                    cwd=solution_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=float(execution["checker_timeout_seconds"]),
                    check=False,
                )
                error = None if checked.returncode == 0 else "wrong_answer"
            except subprocess.TimeoutExpired:
                error = "checker_timeout"
        if error is None:
            passed += 1
        elif first_error is None:
            first_error = error
    return {
        "compiled": True,
        "passed": passed,
        "total": len(tests),
        "pass_rate": passed / len(tests),
        "error": first_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CodeContests+ tests, checkers, and submissions")
    parser.add_argument("--config", default="configs/data/m10_codecontests_plus_audit.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = config["dataset"]
    selection = config["selection"]

    fs = HfFileSystem()
    info = HfApi().dataset_info(dataset["name"], revision=dataset["revision"])
    shards = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if fnmatch(sibling.rfilename, dataset["shard_glob"])
    )[: int(selection["shard_count"])]
    if not shards:
        raise RuntimeError(f"No dataset files matched {dataset['shard_glob']!r}")
    columns = [
        "source", "id", "title", "description", "checker", "correct_submissions",
        "incorrect_submissions", "test_cases", "true_positive_rate", "true_negative_rate",
    ]
    rows = []
    for shard_index, shard in enumerate(shards, 1):
        remote = f"datasets/{dataset['name']}/{shard}"
        print(f"Reading selected columns from shard {shard_index}/{len(shards)}: {remote}", flush=True)
        with fs.open(remote, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            row_groups = min(int(selection["row_groups_per_shard"]), parquet.num_row_groups)
            rows.extend(parquet.read_row_groups(range(row_groups), columns=columns).to_pylist())
    eligible = [
        row for row in rows
        if (row.get("true_positive_rate") or 0) >= selection["minimum_true_positive_rate"]
        and (row.get("true_negative_rate") or 0) >= selection["minimum_true_negative_rate"]
        and row.get("checker")
        and row.get("test_cases")
        and cpp_submissions(row.get("correct_submissions"), selection["language"])
        and cpp_submissions(row.get("incorrect_submissions"), selection["language"])
    ]
    selected = stable_sample(eligible, int(selection["sample_size"]), int(selection["seed"]))
    if len(selected) < int(selection["sample_size"]):
        raise RuntimeError(f"Only {len(selected)} eligible rows found in the selected shard")

    testlib = ensure_testlib(Path(config["testlib"]["cache_path"]), config["testlib"]["revision"])
    results = []
    for position, row in enumerate(selected, 1):
        tests = row["test_cases"][: int(selection["tests_per_problem"])]
        with tempfile.TemporaryDirectory(prefix="qwen3-ccplus-audit-") as directory:
            workdir = Path(directory)
            checker, checker_stderr = compile_checker(
                row["checker"], testlib, workdir, float(config["execution"]["compile_timeout_seconds"])
            )
            correct_results = []
            incorrect_results = []
            if checker is not None:
                for code in cpp_submissions(row["correct_submissions"], selection["language"])[
                    : int(selection["correct_submissions_per_problem"])
                ]:
                    correct_results.append(judge_submission(code, tests, checker, workdir, config["execution"]))
                for code in cpp_submissions(row["incorrect_submissions"], selection["language"])[
                    : int(selection["incorrect_submissions_per_problem"])
                ]:
                    incorrect_results.append(judge_submission(code, tests, checker, workdir, config["execution"]))
            results.append({
                "problem_id": f"{row['source']}:{row['id']}",
                "title": row["title"],
                "tests_used": len(tests),
                "checker_compiled": checker is not None,
                "checker_compile_stderr": checker_stderr,
                "published_true_positive_rate": row["true_positive_rate"],
                "published_true_negative_rate": row["true_negative_rate"],
                "correct_submissions": correct_results,
                "incorrect_submissions": incorrect_results,
            })
        print(f"Audited {position}/{len(selected)}: {row['source']}:{row['id']}", flush=True)

    checker_compiled = sum(row["checker_compiled"] for row in results)
    correct = [attempt for row in results for attempt in row["correct_submissions"]]
    incorrect = [attempt for row in results for attempt in row["incorrect_submissions"]]
    summary = {
        "sampled_problems": len(results),
        "eligible_in_scanned_row_groups": len(eligible),
        "checker_compile_rate": checker_compiled / len(results),
        "correct_submission_accept_rate": sum(x.get("pass_rate") == 1 for x in correct) / len(correct) if correct else None,
        "incorrect_submission_reject_rate": sum(x.get("pass_rate") != 1 for x in incorrect) / len(incorrect) if incorrect else None,
        "correct_errors": dict(Counter(x.get("error") for x in correct)),
        "incorrect_errors": dict(Counter(x.get("error") for x in incorrect)),
    }
    report = {
        "schema_version": "codecontests-plus-audit-v1",
        "dataset": dataset,
        "shards": shards,
        "selection": selection,
        "summary": summary,
        "problems": results,
    }
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "audit_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
