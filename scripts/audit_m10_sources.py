from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.data.agent_eval import read_jsonl, write_jsonl
from src.verifier import judge


def tests_sha256(tests: list[dict[str, str]]) -> str:
    payload = json.dumps(tests, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_tests(value: Any, *, max_tests: int) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_int=Decimal)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(value, dict) or value.get("fn_name"):
        return None
    inputs, outputs = value.get("inputs"), value.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None
    tests: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for input_value, output_value in zip(inputs, outputs):
        if not isinstance(input_value, str) or not isinstance(output_value, str):
            return None
        key = (input_value, output_value)
        if key in seen or len(input_value) > 1_000_000 or len(output_value) > 1_000_000:
            continue
        seen.add(key)
        tests.append({"input": input_value, "output": output_value})
        if len(tests) == max_tests:
            break
    return tests if len(tests) >= 2 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Redownload and audit M10 TACO tests against OCR2 code")
    parser.add_argument("--dataset", default="data/processed/repair_sft_v1/train_agent_pilot.jsonl")
    parser.add_argument("--sft", default="data/processed/sft_10k.jsonl")
    parser.add_argument("--output-dir", default="data/processed/repair_sft_v1/source_audit")
    parser.add_argument("--revision", default="d593ed0a2becbbc952230bb89be09189bf1056dc")
    parser.add_argument("--max-tests", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frozen = read_jsonl(args.dataset)
    by_index = {int(row["metadata"]["test_source_index"]): row for row in frozen}
    if len(by_index) != len(frozen):
        raise ValueError("Duplicate TACO source indices in frozen M10 dataset")

    from datasets import load_dataset

    data_file = f"hf://datasets/BAAI/TACO@{args.revision}/ALL/train-*.parquet"
    stream = load_dataset("parquet", data_files=data_file, split="train", streaming=True)
    fresh: dict[int, list[dict[str, str]]] = {}
    iterator = iter(stream)
    try:
        for index, row in enumerate(iterator):
            if index not in by_index:
                continue
            tests = parse_tests(row.get("input_output"), max_tests=args.max_tests)
            if tests is not None:
                fresh[index] = tests
            if len(fresh) == len(by_index):
                break
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()

    codes: dict[str, str] = {}
    wanted_ids = {row["problem_id"] for row in frozen}
    for row in read_jsonl(args.sft):
        if row["problem_id"] in wanted_ids:
            codes[row["problem_id"]] = row["code"]

    def verify(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        index = int(row["metadata"]["test_source_index"])
        tests = fresh.get(index)
        frozen_hash = tests_sha256(row["tests"])
        fresh_hash = tests_sha256(tests) if tests else None
        result = None
        if tests and row["problem_id"] in codes:
            judged = judge(
                codes[row["problem_id"]],
                tests,
                compile_timeout_seconds=10,
                execution_timeout_seconds=6,
                memory_limit_bytes=512 * 1024 * 1024,
                output_limit_bytes=1024 * 1024,
            )
            result = judged.to_dict()
        return row["problem_id"], {
            "test_source_index": index,
            "fresh_tests_found": tests is not None,
            "fresh_matches_frozen": fresh_hash == frozen_hash,
            "frozen_test_sha256": frozen_hash,
            "fresh_test_sha256": fresh_hash,
            "reference_judge": result,
        }

    audits: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(verify, row) for row in frozen]
        for completed, future in enumerate(as_completed(futures), 1):
            problem_id, audit = future.result()
            audits[problem_id] = audit
            if completed % 25 == 0 or completed == len(futures):
                print(f"Audited {completed}/{len(futures)}", flush=True)

    clean = [
        row
        for row in frozen
        if (audits[row["problem_id"]].get("reference_judge") or {}).get("pass_rate") == 1.0
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train_agent_reference_full_pass.jsonl", clean)
    report = {
        "schema_version": "m10-source-audit-v1",
        "taco_revision": args.revision,
        "hf_home": os.environ.get("HF_HOME"),
        "counts": {
            "frozen_rows": len(frozen),
            "fresh_tests_found": sum(a["fresh_tests_found"] for a in audits.values()),
            "fresh_matches_frozen": sum(a["fresh_matches_frozen"] for a in audits.values()),
            "reference_full_pass": len(clean),
            "reference_not_full_pass": len(frozen) - len(clean),
            "reference_errors": dict(
                collections.Counter(
                    (a.get("reference_judge") or {}).get("error_type") for a in audits.values()
                )
            ),
        },
        "problems": audits,
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["counts"], indent=2))
    print(f"Report: {output / 'audit_report.json'}")
    print(f"Clean dataset: {output / 'train_agent_reference_full_pass.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
