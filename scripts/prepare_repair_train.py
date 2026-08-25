from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.agent_eval import row_test_hash, split_visible_hidden_tests, write_jsonl
from src.data.sft import is_eval_leak, stable_order
from src.utils.config import load_config, require_sections


def parse_taco_tests(value: Any, *, max_tests: int) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or value.get("fn_name"):
        return None
    inputs, outputs = value.get("inputs"), value.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None
    tests = []
    seen = set()
    for input_value, output_value in zip(inputs, outputs):
        if not isinstance(input_value, str) or not isinstance(output_value, str):
            continue
        key = (input_value, output_value)
        if key in seen or len(input_value) > 1_000_000 or len(output_value) > 1_000_000:
            continue
        seen.add(key)
        tests.append({"input": input_value, "output": output_value})
        if len(tests) == max_tests:
            break
    return tests if len(tests) >= 2 else None


def select_candidates(sft_rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    values = [
        row
        for row in sft_rows
        if row.get("metadata", {}).get("dataset") == "taco"
        and row.get("metadata", {}).get("dataset_split") == "train"
    ]
    return stable_order(values, seed)[:size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an eval-isolated TACO repair train split")
    parser.add_argument("--config", default="configs/data/m10_repair_train_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "selection", "test_split", "output")
    cache = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
    if cache.is_dir():
        os.environ.setdefault("HF_HOME", str(cache))
    source = config["source"]
    sft_rows = [json.loads(line) for line in open(source["sft_path"], encoding="utf-8") if line.strip()]
    selection = config["selection"]
    candidates = select_candidates(
        sft_rows,
        size=int(selection["candidate_count"]),
        seed=int(selection["seed"]),
    )
    wanted = {int(row["metadata"]["dataset_index"]): row for row in candidates}
    if len(wanted) != len(candidates):
        raise ValueError("Selected TACO candidates contain duplicate dataset indices")

    from datasets import load_dataset

    pattern = source["data_pattern"].format(split="train")
    data_file = f"hf://datasets/{source['dataset']}@{source['revision']}/{pattern}"
    stream = load_dataset("parquet", data_files=data_file, split="train", streaming=True)
    iterator = iter(stream)
    resolved: dict[int, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    fingerprints = json.loads(Path(source["eval_fingerprints"]).read_text(encoding="utf-8"))[
        "fingerprints"
    ]
    remaining = set(wanted)
    try:
        for index, taco in enumerate(iterator):
            if index not in remaining:
                continue
            selected = wanted[index]
            remaining.remove(index)
            if is_eval_leak(selected["problem"], fingerprints):
                rejected["eval_leak"] += 1
                continue
            tests = parse_taco_tests(
                taco.get("input_output"), max_tests=int(selection["max_tests"])
            )
            if tests is None:
                rejected["unsupported_or_insufficient_tests"] += 1
                continue
            row = {
                "problem_id": selected["problem_id"],
                "source": "BAAI/TACO",
                "problem": selected["problem"],
                "language": "cpp",
                "difficulty": selected["difficulty"],
                "tests": tests,
                "metadata": {
                    **selected.get("metadata", {}),
                    "test_source_dataset": source["dataset"],
                    "test_source_revision": source["revision"],
                    "test_source_index": index,
                },
            }
            adapted = split_visible_hidden_tests(
                row,
                seed=int(config["test_split"]["seed"]),
                visible_fraction=float(config["test_split"]["visible_fraction"]),
                visible_max=int(config["test_split"]["visible_max"]),
            )
            # Unlike evaluation datasets, the one-shot failure producer is judged
            # against every training-side test. visible/hidden remain available for
            # the subsequent repair loop, but neither partition is an eval secret.
            adapted["tests"] = tests
            resolved[index] = adapted
            if len(resolved) >= int(selection["target_count"]):
                break
    finally:
        # datasets streaming iterators may own httpx/pyarrow background state.
        # Closing the generator while Python is still live prevents a known
        # PyGILState_Release crash during interpreter finalization.
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        del iterator, stream
        gc.collect()
    rows = [resolved[index] for index in wanted if index in resolved][
        : int(selection["target_count"])
    ]
    if len(rows) < int(selection["target_count"]):
        raise RuntimeError(
            f"Only {len(rows)} TACO problems survived; need {selection['target_count']}. "
            f"Unresolved source indices: {len(remaining)}"
        )
    output = config["output"]
    write_jsonl(output["dataset"], rows)
    manifest = {
        "schema_version": "repair-train-v1",
        "source": source,
        "selection": selection,
        "test_split": config["test_split"],
        "problem_ids": {"train": [row["problem_id"] for row in rows]},
        "test_sha256": {row["problem_id"]: row_test_hash(row) for row in rows},
        "counts": {
            "problems": len(rows),
            "tests": sum(len(row["tests"]) for row in rows),
            "visible_tests": sum(len(row["visible_tests"]) for row in rows),
            "hidden_tests": sum(len(row["hidden_tests"]) for row in rows),
            "difficulty": dict(Counter(row["difficulty"] for row in rows)),
            "rejected": dict(rejected),
        },
        "dataset_sha256": hashlib.sha256(Path(output["dataset"]).read_bytes()).hexdigest(),
    }
    manifest_path = Path(output["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Some datasets/pyarrow streaming builds abort in a background HTTP
    # finalizer after main has completed successfully. All outputs are closed
    # and hashed above, so bypass only the faulty interpreter finalization in
    # this dedicated subprocess.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
