from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

try:
    from scripts.audit_codecontests_plus import (
        cpp_submissions,
        ensure_testlib,
        prepare_contest_submission,
        stable_sample,
    )
except ModuleNotFoundError:
    from audit_codecontests_plus import (
        cpp_submissions,
        ensure_testlib,
        prepare_contest_submission,
        stable_sample,
    )
from src.data.agent_eval import row_test_hash, split_visible_hidden_tests, write_jsonl
from src.data.sft import is_eval_leak
from src.utils.config import load_config, require_sections
from src.verifier import judge


def source_shards(config: dict[str, Any]) -> list[str]:
    source, selection = config["source"], config["selection"]
    info = HfApi().dataset_info(source["dataset"], revision=source["revision"])
    return sorted(
        sibling.rfilename
        for sibling in info.siblings
        if fnmatch(sibling.rfilename, source["shard_glob"])
    )[: int(source["shard_count"])]


def iter_candidate_rows(
    config: dict[str, Any], shards: list[str]
) -> Any:
    source, selection = config["source"], config["selection"]
    columns = [
        "source", "id", "title", "description", "time_limit", "memory_limit", "checker",
        "correct_submissions", "incorrect_submissions", "test_cases", "true_positive_rate",
        "true_negative_rate",
    ]
    fs = HfFileSystem()
    for index, shard in enumerate(shards, 1):
        print(f"Reading CodeContests+ shard {index}/{len(shards)}: {shard}", flush=True)
        with fs.open(f"datasets/{source['dataset']}/{shard}", "rb") as handle:
            parquet = pq.ParquetFile(handle)
            count = min(int(source["row_groups_per_shard"]), parquet.num_row_groups)
            for row_group in range(count):
                rows = parquet.read_row_group(row_group, columns=columns).to_pylist()
                if source.get("streaming", False):
                    rows = stable_sample(rows, len(rows), int(selection["seed"]))
                print(
                    f"  Loaded row group {row_group + 1}/{count}; "
                    f"rows in memory: {len(rows)}",
                    flush=True,
                )
                yield from rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare checker-backed CodeContests+ repair data")
    parser.add_argument(
        "--config", default="configs/data/m10_codecontests_plus_repair_smoke_v1.yaml"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the unchanged selection gates and report counts without writing datasets.",
    )
    parser.add_argument(
        "--audit-output",
        default="outputs/audit/m11-codecontests-plus-rejected-300.jsonl",
        help="Compact rejected-candidate records written with --audit-only.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Override selection.target_count (useful for a smaller audit sample).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Stop an audit after scanning this many candidates, regardless of accepts.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "selection", "execution", "test_split", "testlib", "output")
    source, selection, execution = config["source"], config["selection"], config["execution"]
    if args.target_count is not None:
        if args.target_count < 1:
            parser.error("--target-count must be positive")
        selection["target_count"] = args.target_count
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    testlib = ensure_testlib(Path(config["testlib"]["cache_path"]), config["testlib"]["revision"])
    shards = source_shards(config)
    if source.get("streaming", False):
        candidates = iter_candidate_rows(config, shards)
    else:
        loaded_candidates = list(iter_candidate_rows(config, shards))
        candidates = stable_sample(
            loaded_candidates, len(loaded_candidates), int(selection["seed"])
        )
    fingerprints = json.loads(Path(source["eval_fingerprints"]).read_text(encoding="utf-8"))[
        "fingerprints"
    ]
    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    one_shot_seeds: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    rejected_records: list[dict[str, Any]] = []
    accepted_count = 0
    scanned_count = 0

    def reject(row: dict[str, Any], candidate_index: int, reason: str, **details: Any) -> None:
        rejected[reason] += 1
        if args.audit_only:
            rejected_records.append({
                "candidate_index": candidate_index,
                "problem_id": f"ccplus:{row.get('source')}:{row.get('id')}",
                "reason": reason,
                "has_checker": bool(row.get("checker")),
                "test_count": len(row.get("test_cases") or []),
                "true_positive_rate": row.get("true_positive_rate"),
                "true_negative_rate": row.get("true_negative_rate"),
                "correct_cpp_submissions": len(
                    cpp_submissions(row.get("correct_submissions"), selection["language"])
                ),
                "incorrect_cpp_submissions": len(
                    cpp_submissions(row.get("incorrect_submissions"), selection["language"])
                ),
                **details,
            })

    for candidate_index, row in enumerate(candidates, 1):
        if args.audit_only and args.max_candidates is not None and candidate_index > args.max_candidates:
            break
        if accepted_count >= int(selection["target_count"]):
            break
        scanned_count = candidate_index
        tests = row.get("test_cases") or []
        if (
            (row.get("true_positive_rate") or 0) < float(selection["minimum_true_positive_rate"])
            or (row.get("true_negative_rate") or 0) < float(selection["minimum_true_negative_rate"])
        ):
            reject(row, candidate_index, "quality_threshold")
            continue
        if not row.get("checker") or len(tests) < int(selection["minimum_tests"]):
            reject(row, candidate_index, "missing_checker_or_tests")
            continue
        if is_eval_leak(row["description"], fingerprints):
            reject(row, candidate_index, "eval_leak")
            continue
        correct_codes = cpp_submissions(row.get("correct_submissions"), selection["language"])
        incorrect_codes = cpp_submissions(row.get("incorrect_submissions"), selection["language"])
        if not correct_codes or not incorrect_codes:
            reject(row, candidate_index, "missing_cpp_submission_class")
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
            reject(
                row,
                candidate_index,
                f"correct_gate_{correct_judge.error_type or 'not_full_pass'}",
                correct_judge={
                    "compiled": correct_judge.compiled,
                    "pass_rate": correct_judge.pass_rate,
                    "error_type": correct_judge.error_type,
                },
            )
            continue
        initial_code = incorrect_codes[0]
        initial_judge = judge(prepare_contest_submission(initial_code), tests, **judge_args)
        if initial_judge.pass_rate == 1:
            reject(
                row,
                candidate_index,
                "incorrect_gate_full_pass",
                incorrect_judge={
                    "compiled": initial_judge.compiled,
                    "pass_rate": initial_judge.pass_rate,
                    "error_type": initial_judge.error_type,
                },
            )
            continue

        problem_id = f"ccplus:{row['source']}:{row['id']}"
        accepted_count += 1
        if args.audit_only:
            print(
                f"Accepted {accepted_count}/{selection['target_count']} after {candidate_index} "
                f"candidates: {problem_id} ({initial_judge.error_type})",
                flush=True,
            )
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
        # Producers and final gates consume every training-side test.
        adapted["tests"] = tests
        accepted.append(adapted)
        task_id = hashlib.sha256(f"ccplus-repair-v1:{problem_id}".encode()).hexdigest()
        failures.append({
            "task_id": task_id,
            "problem": adapted,
            "initial_submission": {
                "producer_model": "CodeContests+ authentic incorrect submission",
                "sample_index": 0,
                "response": f"```cpp\n{initial_code.strip()}\n```",
                "code": prepare_contest_submission(initial_code),
                "finish_reason": "dataset",
                "generation_tokens": 0,
                "source_judge": initial_judge.to_dict(),
            },
        })
        one_shot_seeds.append(
            {
                "schema_version": "checker-backed-one-shot-seed-v1",
                "problem_id": problem_id,
                "response": f"```cpp\n{correct_code.strip()}\n```",
                "code": prepare_contest_submission(correct_code),
                "source_judge": correct_judge.to_dict(),
            }
        )
        print(
            f"Accepted {accepted_count}/{selection['target_count']} after {candidate_index} candidates: "
            f"{problem_id} ({initial_judge.error_type})",
            flush=True,
        )

    bounded_audit = args.audit_only and args.max_candidates is not None
    if not bounded_audit and accepted_count < int(selection["target_count"]):
        raise RuntimeError(f"Only {accepted_count} problems survived; need {selection['target_count']}")
    if args.audit_only:
        write_jsonl(args.audit_output, rejected_records)
        summary = {
            "scanned_candidates": scanned_count,
            "accepted": accepted_count,
            "accept_rate": accepted_count / scanned_count if scanned_count else 0.0,
            "rejected": dict(rejected),
            "rejected_records": args.audit_output,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    output = config["output"]
    write_jsonl(output["dataset"], accepted)
    write_jsonl(output["failure_pool"], failures)
    if output.get("one_shot_seeds"):
        write_jsonl(output["one_shot_seeds"], one_shot_seeds)
    manifest = {
        "schema_version": "codecontests-plus-repair-v1",
        "source": source,
        "shards": shards,
        "selection": selection,
        "execution": execution,
        "test_split": config["test_split"],
        "problem_ids": {"train": [row["problem_id"] for row in accepted]},
        "test_sha256": {row["problem_id"]: row_test_hash(row) for row in accepted},
        "counts": {
            "problems": len(accepted),
            "failure_tasks": len(failures),
            "one_shot_seeds": len(one_shot_seeds),
            "tests": sum(len(row["tests"]) for row in accepted),
            "visible_tests": sum(len(row["visible_tests"]) for row in accepted),
            "hidden_tests": sum(len(row["hidden_tests"]) for row in accepted),
            "initial_error_types": dict(
                Counter(row["initial_submission"]["source_judge"]["error_type"] for row in failures)
            ),
            "rejected": dict(rejected),
        },
    }
    dataset_path = Path(output["dataset"])
    manifest["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if output.get("one_shot_seeds"):
        manifest["one_shot_seeds_sha256"] = hashlib.sha256(
            Path(output["one_shot_seeds"]).read_bytes()
        ).hexdigest()
    manifest_path = Path(output["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
