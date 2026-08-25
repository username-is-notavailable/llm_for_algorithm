from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    from scripts.audit_taco_native_solutions import judge_python, parse_solutions
    from scripts.prepare_repair_train import parse_taco_tests
except ModuleNotFoundError:  # Direct execution adds scripts/, not the project root, to sys.path.
    from audit_taco_native_solutions import judge_python, parse_solutions
    from prepare_repair_train import parse_taco_tests
from src.data.agent_eval import row_test_hash, split_visible_hidden_tests, write_jsonl
from src.data.sft import is_eval_leak, normalize_difficulty
from src.utils.config import load_config, require_sections


def problem_id(revision: str, index: int, question: str) -> str:
    digest = hashlib.sha256(f"{revision}\0{index}\0{question.strip()}".encode()).hexdigest()
    return f"taco:{digest}"


def candidate_digest(seed: int, index: int, question: str) -> bytes:
    return hashlib.sha256(f"{seed}:{index}:{question.strip()}".encode()).digest()


def is_unsupported(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key) or "") for key in ("question", "raw_tags", "tags"))
    return "interactive" in text.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TACO-native validated Agent SFT smoke data")
    parser.add_argument("--config", default="configs/data/m10_taco_native_sft_smoke_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "selection", "native_gate", "test_split", "output")
    source, selection = config["source"], config["selection"]
    fingerprints = json.loads(Path(source["eval_fingerprints"]).read_text(encoding="utf-8"))[
        "fingerprints"
    ]

    from datasets import load_dataset

    pattern = source["data_pattern"].format(split="train")
    data_file = f"hf://datasets/{source['dataset']}@{source['revision']}/{pattern}"
    stream = load_dataset("parquet", data_files=data_file, split="train", streaming=True)
    iterator = iter(stream)
    candidates: list[dict[str, Any]] = []
    rejected: collections.Counter[str] = collections.Counter()
    try:
        for index, taco in enumerate(iterator):
            if index >= int(selection["scan_limit"]):
                break
            question = taco.get("question")
            if not isinstance(question, str) or len(question.strip()) < 80:
                rejected["missing_or_short_question"] += 1
                continue
            if is_unsupported(taco):
                rejected["interactive"] += 1
                continue
            tests = parse_taco_tests(taco.get("input_output"), max_tests=int(selection["max_tests"]))
            if tests is None:
                rejected["unsupported_or_insufficient_tests"] += 1
                continue
            solutions = parse_solutions(taco.get("solutions"))
            if not solutions:
                rejected["missing_native_solution"] += 1
                continue
            if is_eval_leak(question, fingerprints):
                rejected["eval_leak"] += 1
                continue
            candidates.append(
                {
                    "index": index,
                    "question": question.strip(),
                    "tests": tests,
                    "solutions": solutions,
                    "difficulty": normalize_difficulty(taco.get("difficulty")),
                    "platform": taco.get("source"),
                    "url": taco.get("url"),
                }
            )
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
        del iterator, stream
        gc.collect()

    candidates.sort(
        key=lambda row: candidate_digest(int(selection["seed"]), row["index"], row["question"])
    )
    candidates = candidates[: int(selection["candidate_pool_size"])]
    accepted = []
    gate = config["native_gate"]
    attempted_solutions = 0
    for checked, candidate in enumerate(candidates, 1):
        passing_index = None
        passing_result = None
        for solution_index, solution in enumerate(
            candidate["solutions"][: int(selection["max_native_solutions"])]
        ):
            attempted_solutions += 1
            result = judge_python(
                solution,
                candidate["tests"],
                timeout_seconds=float(gate["execution_timeout_seconds"]),
                memory_limit_bytes=int(gate["memory_limit_mb"]) * 1024 * 1024,
                output_limit_bytes=int(gate["output_limit_bytes"]),
                stop_on_first_failure=True,
            )
            if result["pass_rate"] == 1.0:
                passing_index, passing_result = solution_index, result
                break
        if passing_index is None:
            rejected["native_solution_not_full_pass"] += 1
        else:
            identifier = problem_id(source["revision"], candidate["index"], candidate["question"])
            row = {
                "problem_id": identifier,
                "source": source["dataset"],
                "problem": candidate["question"],
                "language": "cpp",
                "difficulty": candidate["difficulty"],
                "tests": candidate["tests"],
                "metadata": {
                    "dataset": "taco",
                    "dataset_split": "train",
                    "dataset_index": candidate["index"],
                    "platform": candidate["platform"],
                    "url": candidate["url"],
                    "test_source_dataset": source["dataset"],
                    "test_source_revision": source["revision"],
                    "test_source_index": candidate["index"],
                    "native_verification": {
                        "language": "python",
                        "solution_index": passing_index,
                        "solution_sha256": hashlib.sha256(
                            candidate["solutions"][passing_index].encode()
                        ).hexdigest(),
                        **passing_result,
                    },
                },
            }
            adapted = split_visible_hidden_tests(
                row,
                seed=int(config["test_split"]["seed"]),
                visible_fraction=float(config["test_split"]["visible_fraction"]),
                visible_max=int(config["test_split"]["visible_max"]),
            )
            adapted["tests"] = candidate["tests"]
            accepted.append(adapted)
        if checked % 25 == 0:
            print(
                f"Native checked {checked}/{len(candidates)}; accepted "
                f"{len(accepted)}/{selection['target_count']}",
                flush=True,
            )
        if len(accepted) == int(selection["target_count"]):
            break

    if len(accepted) < int(selection["target_count"]):
        raise RuntimeError(
            f"Only {len(accepted)} native-verified problems survived; "
            f"need {selection['target_count']}"
        )
    output = config["output"]
    write_jsonl(output["dataset"], accepted)
    manifest = {
        "schema_version": "repair-train-taco-native-v1",
        "source": source,
        "selection": selection,
        "native_gate": config["native_gate"],
        "test_split": config["test_split"],
        "problem_ids": {"train": [row["problem_id"] for row in accepted]},
        "test_sha256": {row["problem_id"]: row_test_hash(row) for row in accepted},
        "counts": {
            "problems": len(accepted),
            "tests": sum(len(row["tests"]) for row in accepted),
            "visible_tests": sum(len(row["visible_tests"]) for row in accepted),
            "hidden_tests": sum(len(row["hidden_tests"]) for row in accepted),
            "difficulty": dict(collections.Counter(row["difficulty"] for row in accepted)),
            "platform": dict(
                collections.Counter(row["metadata"].get("platform") for row in accepted)
            ),
            "candidate_pool": len(candidates),
            "native_solutions_attempted": attempted_solutions,
            "rejected": dict(rejected),
        },
    }
    manifest_path = Path(output["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["dataset_sha256"] = hashlib.sha256(Path(output["dataset"]).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit as error:
        exit_code = int(error.code or 0)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
