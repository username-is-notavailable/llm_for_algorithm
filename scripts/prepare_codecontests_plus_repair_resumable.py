from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

from scripts.audit_codecontests_plus import cpp_submissions, ensure_testlib, prepare_contest_submission
from scripts.prepare_codecontests_plus_repair import (
    excluded_problem_ids,
    iter_candidate_rows,
    source_shards,
)
from src.data.agent_eval import row_test_hash, split_visible_hidden_tests
from src.data.sft import is_eval_leak
from src.utils.config import load_config, require_sections
from src.verifier import judge


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def judge_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "compiled", "passed", "total", "pass_rate", "runtime_error", "timeout", "error_type"
        )
        if key in value
    }


def write_record(handle: BinaryIO, row: dict[str, Any]) -> tuple[int, int]:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    offset = handle.tell()
    handle.write(payload)
    return offset, len(payload)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def initial_state(config: dict[str, Any], files: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": "codecontests-plus-repair-checkpoint-v1",
        "config_sha256": config_digest(config),
        "complete": False,
        "scanned_candidates": 0,
        "accepted": 0,
        "rejected": {},
        "records": {},
        "counts": {"tests": 0, "visible_tests": 0, "hidden_tests": 0},
        "initial_error_types": {},
        "file_sizes": {name: 0 for name in files},
    }


def sync_checkpoint(
    state_path: Path,
    state: dict[str, Any],
    handles: dict[str, BinaryIO],
) -> None:
    for name, handle in handles.items():
        handle.flush()
        os.fsync(handle.fileno())
        state["file_sizes"][name] = handle.tell()
    atomic_write_json(state_path, state)


def open_outputs(
    config: dict[str, Any], *, resume: bool
) -> tuple[dict[str, Any], dict[str, BinaryIO], dict[str, Path], Path]:
    output = config["output"]
    files = {
        "problems": Path(output["dataset"]),
        "failure_pool": Path(output["failure_pool"]),
        "one_shot_seeds": Path(output["one_shot_seeds"]),
    }
    state_path = Path(output["checkpoint"])
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config_sha256"] != config_digest(config):
            raise ValueError("Checkpoint config differs from the requested config")
        if state.get("complete"):
            raise ValueError("Checkpoint is already complete")
        handles = {name: path.open("a+b") for name, path in files.items()}
        for name, handle in handles.items():
            committed = int(state["file_sizes"][name])
            handle.truncate(committed)
            handle.seek(committed)
        print(
            f"Resuming after candidate {state['scanned_candidates']} with "
            f"{state['accepted']} accepted rows",
            flush=True,
        )
        return state, handles, files, state_path

    existing = [path for path in [state_path, *files.values()] if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing partial data; rerun with --resume or remove: "
            + ", ".join(map(str, existing))
        )
    handles = {name: path.open("w+b") for name, path in files.items()}
    state = initial_state(config, files)
    sync_checkpoint(state_path, state, handles)
    return state, handles, files, state_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable, streaming CodeContests+ repair preparation")
    parser.add_argument("--config", default="configs/data/m12_codecontests_plus_repair_1000_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "selection", "execution", "test_split", "testlib", "output")
    source, selection, execution = config["source"], config["selection"], config["execution"]
    testlib = ensure_testlib(Path(config["testlib"]["cache_path"]), config["testlib"]["revision"])
    shards = source_shards(config)
    fingerprints = json.loads(Path(source["eval_fingerprints"]).read_text(encoding="utf-8"))["fingerprints"]
    excluded_ids = excluded_problem_ids(selection)
    state, handles, files, state_path = open_outputs(config, resume=args.resume)
    rejected = Counter(state["rejected"])
    error_types = Counter(state["initial_error_types"])
    counts = Counter(state["counts"])
    accepted_count = int(state["accepted"])
    resume_after = int(state["scanned_candidates"])
    target_count = int(selection["target_count"])
    print(f"Excluding {len(excluded_ids)} frozen problem IDs", flush=True)

    def checkpoint(candidate_index: int, *, sync_files: bool) -> None:
        state.update(
            {
                "scanned_candidates": candidate_index,
                "accepted": accepted_count,
                "rejected": dict(rejected),
                "counts": dict(counts),
                "initial_error_types": dict(error_types),
            }
        )
        if sync_files:
            sync_checkpoint(state_path, state, handles)
        else:
            atomic_write_json(state_path, state)

    try:
        for candidate_index, row in enumerate(iter_candidate_rows(config, shards), 1):
            if candidate_index <= resume_after:
                continue
            if accepted_count >= target_count:
                break
            problem_id = f"ccplus:{row.get('source')}:{row.get('id')}"
            reason = None
            tests = row.get("test_cases") or []
            if problem_id in excluded_ids:
                reason = "excluded_existing_problem"
            elif (
                (row.get("true_positive_rate") or 0) < float(selection["minimum_true_positive_rate"])
                or (row.get("true_negative_rate") or 0) < float(selection["minimum_true_negative_rate"])
            ):
                reason = "quality_threshold"
            elif not row.get("checker") or len(tests) < int(selection["minimum_tests"]):
                reason = "missing_checker_or_tests"
            elif is_eval_leak(row["description"], fingerprints):
                reason = "eval_leak"
            correct_codes = cpp_submissions(row.get("correct_submissions"), selection["language"])
            incorrect_codes = cpp_submissions(row.get("incorrect_submissions"), selection["language"])
            if reason is None and (not correct_codes or not incorrect_codes):
                reason = "missing_cpp_submission_class"
            if reason is not None:
                rejected[reason] += 1
                checkpoint(candidate_index, sync_files=False)
                continue

            judge_args = {
                "compile_timeout_seconds": float(execution["compile_timeout_seconds"]),
                "execution_timeout_seconds": float(execution["execution_timeout_seconds"]),
                "memory_limit_bytes": int(execution["memory_limit_bytes"]),
                "output_limit_bytes": int(execution["output_limit_bytes"]),
                "output_checker_source": row["checker"],
                "testlib_path": testlib,
                "checker_timeout_seconds": float(execution["checker_timeout_seconds"]),
            }
            correct_code = correct_codes[0]
            correct_judge = judge(prepare_contest_submission(correct_code), tests, **judge_args)
            if correct_judge.pass_rate != 1:
                rejected[f"correct_gate_{correct_judge.error_type or 'not_full_pass'}"] += 1
                checkpoint(candidate_index, sync_files=False)
                continue
            initial_code = incorrect_codes[0]
            initial_judge = judge(prepare_contest_submission(initial_code), tests, **judge_args)
            if initial_judge.pass_rate == 1:
                rejected["incorrect_gate_full_pass"] += 1
                checkpoint(candidate_index, sync_files=False)
                continue

            problem = {
                "problem_id": problem_id,
                "source": source["dataset"],
                "problem": row["description"],
                "language": "cpp",
                "difficulty": "unknown",
                "tests": tests,
                "metadata": {
                    "upstream_source": row["source"],
                    "upstream_id": row["id"],
                    "title": row["title"],
                    "dataset_revision": source["revision"],
                    "published_true_positive_rate": row["true_positive_rate"],
                    "published_true_negative_rate": row["true_negative_rate"],
                    "time_limit_ms": row["time_limit"],
                    "memory_limit_mb": row["memory_limit"],
                    "output_checker": {
                        "kind": "testlib-cpp",
                        "source": row["checker"],
                        "testlib_revision": config["testlib"]["revision"],
                        "testlib_path": config["testlib"]["cache_path"],
                        "timeout_seconds": execution["checker_timeout_seconds"],
                    },
                    "verified_correct_code_sha256": hashlib.sha256(correct_code.encode()).hexdigest(),
                },
            }
            adapted = split_visible_hidden_tests(
                problem,
                seed=int(config["test_split"]["seed"]),
                visible_fraction=float(config["test_split"]["visible_fraction"]),
                visible_max=int(config["test_split"]["visible_max"]),
            )
            adapted["tests"] = tests
            compact_problem = dict(adapted)
            compact_problem.pop("tests")
            offset, length = write_record(handles["problems"], compact_problem)
            state["records"][problem_id] = [offset, length]
            task_id = hashlib.sha256(f"ccplus-repair-v1:{problem_id}".encode()).hexdigest()
            initial = {
                "producer_model": "CodeContests+ authentic incorrect submission",
                "sample_index": 0,
                "response": f"```cpp\n{initial_code.strip()}\n```",
                "code": prepare_contest_submission(initial_code),
                "finish_reason": "dataset",
                "generation_tokens": 0,
                "source_judge": judge_summary(initial_judge.to_dict()),
            }
            write_record(
                handles["failure_pool"],
                {
                    "schema_version": "checker-backed-repair-task-v2",
                    "task_id": task_id,
                    "problem_id": problem_id,
                    "initial_submission": initial,
                },
            )
            write_record(
                handles["one_shot_seeds"],
                {
                    "schema_version": "checker-backed-one-shot-seed-v2",
                    "problem_id": problem_id,
                    "response": f"```cpp\n{correct_code.strip()}\n```",
                    "code": prepare_contest_submission(correct_code),
                    "source_judge": judge_summary(correct_judge.to_dict()),
                },
            )
            accepted_count += 1
            counts["tests"] += len(tests)
            counts["visible_tests"] += len(adapted["visible_tests"])
            counts["hidden_tests"] += len(adapted["hidden_tests"])
            error_types[str(initial_judge.error_type)] += 1
            checkpoint(candidate_index, sync_files=True)
            print(
                f"Accepted {accepted_count}/{target_count} after {candidate_index} candidates: "
                f"{problem_id} ({initial_judge.error_type})",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    if accepted_count < target_count:
        raise RuntimeError(f"Only {accepted_count} problems survived; need {target_count}")

    index_path = Path(config["output"]["problem_index"])
    atomic_write_json(
        index_path,
        {"schema_version": "jsonl-byte-offset-index-v1", "records": state["records"]},
    )
    state["complete"] = True
    atomic_write_json(state_path, state)
    manifest = {
        "schema_version": "codecontests-plus-repair-compact-v3",
        "source": source,
        "shards": shards,
        "selection": selection,
        "execution": execution,
        "test_split": config["test_split"],
        "counts": {
            "problems": accepted_count,
            "failure_tasks": accepted_count,
            "one_shot_seeds": accepted_count,
            **dict(counts),
            "initial_error_types": dict(error_types),
            "rejected": dict(rejected),
            "excluded_problem_ids": len(excluded_ids),
            "scanned_candidates": state["scanned_candidates"],
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in [*files.values(), index_path]
        },
    }
    manifest_path = Path(config["output"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
