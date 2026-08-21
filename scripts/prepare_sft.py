from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from src.data.livecodebench import write_jsonl
from src.data.sft import adapt_ocr2, balanced_order, deduplicate_candidates, is_eval_leak
from src.inference.prompts import build_code_prompt
from src.utils.config import load_config, require_sections
from src.verifier.compiler import compile_code


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ocr_rows(config: dict[str, Any], source_jsonl: str | None) -> Iterable[dict[str, Any]]:
    if source_jsonl:
        yield from _load_jsonl(source_jsonl)
        return
    from datasets import load_dataset

    source = config["source"]
    stream = load_dataset(
        source["dataset"],
        split=source["split"],
        revision=source["revision"],
        streaming=True,
    )
    for index, row in enumerate(stream):
        if index >= int(source["candidate_scan_limit"]):
            break
        if index and index % 10000 == 0:
            print(f"Scanned {index} OCR2 rows", flush=True)
        yield row


def _question_text(dataset_name: str, row: dict[str, Any]) -> str | None:
    if dataset_name in {"taco", "apps"}:
        return row.get("question")
    if dataset_name == "code_contests":
        return row.get("description")
    if dataset_name == "open-r1/codeforces":
        description = row.get("description")
        if not description:
            return None
        parts = [description]
        for title, field in (("Input", "input_format"), ("Output", "output_format")):
            if row.get(field):
                parts.extend((title, row[field]))
        if row.get("examples"):
            parts.append("Examples")
            for example in row["examples"]:
                if example.get("input") is not None:
                    parts.extend(("Input", example["input"]))
                if example.get("output") is not None:
                    parts.extend(("Output", example["output"]))
        if row.get("note"):
            parts.extend(("Note", row["note"]))
        return "\n\n".join(parts)
    return None


def _resolve_questions(
    candidates: list[dict[str, Any]], config: dict[str, Any], questions_jsonl: str | None
) -> dict[tuple[str, str, int], str]:
    wanted: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in candidates:
        wanted[(row["dataset"], row["split"])].add(int(row["index"]))
    if questions_jsonl:
        result = {}
        for row in _load_jsonl(questions_jsonl):
            result[(row["dataset"], row["split"], int(row["index"]))] = row["question"]
        return result
    from datasets import load_dataset

    result: dict[tuple[str, str, int], str] = {}
    for (dataset_name, split), indices in sorted(wanted.items()):
        print(f"Resolving {len(indices)} questions from {dataset_name}/{split}", flush=True)
        source = config["question_sources"][dataset_name]
        if source.get("data_pattern"):
            data_file = (
                f"hf://datasets/{source['dataset']}@{source['revision']}/"
                f"{source['data_pattern'].format(split=split)}"
            )
            stream = load_dataset("parquet", data_files=data_file, split="train", streaming=True)
        else:
            stream = load_dataset(
                source["dataset"],
                split=split,
                revision=source["revision"],
                streaming=True,
            )
        remaining = set(indices)
        for index, row in enumerate(stream):
            if index in remaining:
                question = _question_text(dataset_name, row)
                if question:
                    result[(dataset_name, split, index)] = question
                remaining.remove(index)
                if not remaining:
                    break
        print(
            f"Resolved {len(indices) - len(remaining)}/{len(indices)} from {dataset_name}/{split}",
            flush=True,
        )
    return result


def _compiles(row: dict[str, Any]) -> bool:
    with tempfile.TemporaryDirectory(prefix="qwen3-sft-compile-") as directory:
        return compile_code(row["code"], Path(directory), timeout_seconds=10).success


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _statistics(rows: list[dict[str, Any]], reports: dict[str, Any]) -> dict[str, Any]:
    fields = ("problem", "prompt", "reasoning", "code", "response", "total")
    lengths = {field: [row["token_counts"][field] for row in rows] for field in fields}
    return {
        "samples": len(rows),
        "unique_problems": len({row["problem_id"] for row in rows}),
        "verified": Counter(str(row["verified"]).lower() for row in rows),
        "difficulty": Counter(row["difficulty"] for row in rows),
        "platform": Counter(str(row["metadata"].get("platform")) for row in rows),
        "dataset": Counter(str(row["metadata"].get("dataset")) for row in rows),
        "token_percentiles": {
            field: {name: _percentile(values, value) for name, value in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1.0))}
            for field, values in lengths.items()
        },
        "response_over_limit": {
            str(limit): sum(value > limit for value in lengths["response"])
            for limit in (2048, 4096, 8192)
        },
        "total_over_limit": {
            str(limit): sum(value > limit for value in lengths["total"])
            for limit in (4096, 8192, 16384)
        },
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare nested OpenCodeReasoning-2 C++ SFT datasets")
    parser.add_argument("--config", default="configs/data/sft_v1.yaml")
    parser.add_argument("--source-jsonl")
    parser.add_argument("--questions-jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "question_sources", "filter", "tokenizer", "output")
    project_cache = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
    if project_cache.is_dir():
        os.environ.setdefault("HF_HOME", str(project_cache))

    candidates, rejected = deduplicate_candidates(_ocr_rows(config, args.source_jsonl), config["filter"])
    pool_size = int(config["source"]["candidate_pool_size"])
    candidates = balanced_order(candidates, int(config["filter"]["seed"]))[:pool_size]
    questions = _resolve_questions(candidates, config, args.questions_jsonl)
    fingerprints = json.loads(Path(config["output"]["eval_fingerprints"]).read_text())["fingerprints"]
    accepted = []
    for row in candidates:
        key = (row["dataset"], row["split"], int(row["index"]))
        question = questions.get(key)
        if not question or len(question.strip()) < int(config["filter"]["minimum_problem_characters"]):
            rejected["missing_or_short_question"] += 1
            continue
        try:
            sample = adapt_ocr2(row, question)
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        leak = is_eval_leak(sample["problem"], fingerprints)
        if leak:
            rejected[leak] += 1
            continue
        accepted.append(sample)

    target_size = max(int(value) for value in config["output"]["sizes"])
    ordered = balanced_order(accepted, int(config["filter"]["seed"]))
    compile_pool = ordered[: max(target_size, int(target_size * 1.25))]
    if config["filter"]["compile_check"]:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            compile_results = executor.map(_compiles, compile_pool)
            compiled = []
            for index, (row, success) in enumerate(zip(compile_pool, compile_results), start=1):
                if success:
                    compiled.append(row)
                else:
                    rejected["compile_error"] += 1
                if index % 500 == 0:
                    print(f"Compile checked {index}/{len(compile_pool)}", flush=True)
    else:
        compiled = compile_pool
    selected = compiled[:target_size]
    if len(selected) < target_size:
        raise RuntimeError(f"Only {len(selected)} samples survived; need {target_size}")

    from transformers import AutoTokenizer

    tokenizer_config = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["name_or_path"], revision=tokenizer_config["revision"])
    for index, row in enumerate(selected, start=1):
        row["prompt"] = build_code_prompt(row["problem"])
        counts = {
            field: len(tokenizer.encode(row[field], add_special_tokens=False))
            for field in ("problem", "prompt", "reasoning", "code", "response")
        }
        counts["total"] = len(tokenizer.encode(row["prompt"] + row["response"], add_special_tokens=False))
        row["token_counts"] = counts
        if index % 1000 == 0:
            print(f"Tokenized {index}/{len(selected)}", flush=True)

    output = config["output"]
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for size in sorted(int(value) for value in output["sizes"]):
        write_jsonl(output_dir / f"sft_{size // 1000}k.jsonl", selected[:size])
    audit = balanced_order(selected, int(config["filter"]["seed"]) + 1)[:100]
    write_jsonl(output["audit_sample"], audit)
    reports = {
        "rejected": dict(rejected),
        "candidate_pool": len(candidates),
        "question_resolved": len(questions),
        "source": config["source"],
        "question_sources": config["question_sources"],
        "filter": config["filter"],
        "tokenizer": config["tokenizer"],
    }
    stats = _statistics(selected, reports)
    Path(output["stats"]).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "sft_contamination_report.json").write_text(
        json.dumps({"eval_matches_retained": 0, "rejected": {key: value for key, value in rejected.items() if key.startswith("eval_")}}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "sft_dedup_report.json").write_text(
        json.dumps({"unique_problem_ids": len(selected), "duplicates_rejected": rejected["duplicate_problem"]}, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
