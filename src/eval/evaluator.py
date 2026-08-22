from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.eval.metrics import compute_metrics
from src.inference.generate import TextGenerator, create_generator
from src.inference.prompts import create_prompt_builder
from src.utils.config import load_config, require_sections
from src.utils.experiment import collect_environment
from src.utils.reproducibility import set_seed
from src.verifier import extract_code, judge


def load_problems(path: str | Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            problem_id = value.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(f"Line {line_number}: missing problem_id")
            if problem_id in seen_ids:
                raise ValueError(f"Line {line_number}: duplicate problem_id {problem_id}")
            if not isinstance(value.get("problem"), str) or not value["problem"]:
                raise ValueError(f"Line {line_number}: missing problem")
            if value.get("language") != "cpp":
                raise ValueError(f"Line {line_number}: language must be cpp")
            tests = value.get("tests")
            if not isinstance(tests, list) or not tests:
                raise ValueError(f"Line {line_number}: tests must be a non-empty list")
            for test in tests:
                if not isinstance(test, dict) or not isinstance(test.get("input"), str) or not isinstance(
                    test.get("output"), str
                ):
                    raise ValueError(f"Line {line_number}: invalid test case")
            seen_ids.add(problem_id)
            problems.append(value)
    if not problems:
        raise ValueError("Evaluation dataset is empty")
    return problems


def validate_split_manifest(problems: list[dict[str, Any]], path: str | Path, split: str) -> None:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        expected_ids = manifest["problem_ids"][split]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Manifest does not define split {split}") from error
    actual_ids = [problem["problem_id"] for problem in problems]
    if actual_ids != expected_ids:
        raise ValueError(f"Dataset IDs do not match frozen manifest split {split}")


def _create_output_dir(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    timestamp = os.environ.get("EVAL_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(experiment["output_dir"]) / f"{experiment['name']}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def configure_shard(config: dict[str, Any], shard_index: int, num_shards: int) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    sharded = copy.deepcopy(config)
    sharded["experiment"]["name"] += f"-shard-{shard_index + 1:02d}-of-{num_shards:02d}"
    sharded["dataset"]["shard"] = {"index": shard_index, "count": num_shards}
    return sharded


def _load_existing_records(
    path: Path, problems: list[dict[str, Any]], num_samples: int
) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.is_file():
        return [], set()
    known_ids = {problem["problem_id"] for problem in problems}
    records = []
    samples_by_problem: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row.get("problem_id")
            sample_index = row.get("sample_index")
            if problem_id not in known_ids:
                raise ValueError(f"Resume line {line_number}: unknown problem_id {problem_id}")
            if not isinstance(sample_index, int) or not 0 <= sample_index < num_samples:
                raise ValueError(f"Resume line {line_number}: invalid sample_index")
            samples = samples_by_problem.setdefault(problem_id, set())
            if sample_index in samples:
                raise ValueError(f"Resume line {line_number}: duplicate generation")
            samples.add(sample_index)
            records.append(row)
    partial = {
        problem_id: len(samples)
        for problem_id, samples in samples_by_problem.items()
        if len(samples) != num_samples
    }
    if partial:
        raise ValueError(f"Resume file contains partially completed problems: {partial}")
    return records, set(samples_by_problem)


def _prepare_output_dir(config: dict[str, Any], resume: str | Path | None) -> Path:
    if resume is None:
        output_dir = _create_output_dir(config)
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (output_dir / "environment.json").write_text(
            json.dumps(collect_environment(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output_dir
    output_dir = Path(resume)
    if not output_dir.is_dir():
        raise ValueError(f"Resume directory does not exist: {output_dir}")
    saved_config_path = output_dir / "config.yaml"
    if not saved_config_path.is_file() or yaml.safe_load(saved_config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume config does not match the saved experiment config")
    return output_dir


def evaluate(
    config: dict[str, Any], generator: TextGenerator, *, resume: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    require_sections(config, "experiment", "model", "prompt", "dataset", "generation", "verifier")
    set_seed(int(config["experiment"]["seed"]))
    build_prompt = create_prompt_builder(config["prompt"], config["model"])
    problems = load_problems(config["dataset"]["path"])
    manifest_path = config["dataset"].get("manifest")
    if manifest_path:
        manifest_split = config["dataset"].get("manifest_split")
        if not isinstance(manifest_split, str) or not manifest_split:
            raise ValueError("dataset.manifest_split is required with dataset.manifest")
        validate_split_manifest(problems, manifest_path, manifest_split)
    limit = config["dataset"].get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("dataset.limit must be a positive integer")
        problems = problems[:limit]
    shard = config["dataset"].get("shard")
    if shard is not None:
        shard_index = int(shard["index"])
        num_shards = int(shard["count"])
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("Invalid dataset.shard index/count")
        problems = problems[shard_index::num_shards]
        if not problems:
            raise ValueError("Evaluation shard is empty")

    num_samples = int(config["generation"].get("num_samples", 1))
    if num_samples < 1:
        raise ValueError("generation.num_samples must be at least 1")
    generation_options = {
        key: value for key, value in config["generation"].items() if key != "num_samples"
    }
    request_batch_size = int(config.get("inference", {}).get("request_batch_size", 1))
    if request_batch_size < 1:
        raise ValueError("inference.request_batch_size must be at least 1")
    output_dir = _prepare_output_dir(config, resume)
    verifier_config = config["verifier"]
    generations_path = output_dir / "generations.jsonl"
    records, completed_ids = _load_existing_records(generations_path, problems, num_samples)
    pending = [problem for problem in problems if problem["problem_id"] not in completed_ids]
    if completed_ids:
        print(f"Resuming with {len(completed_ids)}/{len(problems)} problems complete", flush=True)
    with generations_path.open("a", encoding="utf-8") as output:
        for batch_start in range(0, len(pending), request_batch_size):
            batch = pending[batch_start : batch_start + request_batch_size]
            completed_before = len(completed_ids)
            print(
                f"[{completed_before + 1}-{completed_before + len(batch)}/{len(problems)}] "
                f"Generating batch of {len(batch)} problems",
                flush=True,
            )
            prompts = [build_prompt(problem["problem"]) for problem in batch]
            batch_responses = generator.generate_batch(
                prompts,
                num_samples=num_samples,
                generation=generation_options,
            )
            if len(batch_responses) != len(batch):
                raise RuntimeError("Generator returned an unexpected number of requests")
            for problem, prompt, responses in zip(batch, prompts, batch_responses):
                if len(responses) != num_samples:
                    raise RuntimeError("Generator returned an unexpected number of samples")
                for sample_index, generated in enumerate(responses):
                    response = generated.text
                    code = extract_code(response)
                    judgement = None
                    if code is not None:
                        judgement = judge(
                            code,
                            problem["tests"],
                            compile_timeout_seconds=float(verifier_config["compile_timeout_seconds"]),
                            execution_timeout_seconds=float(verifier_config["execution_timeout_seconds"]),
                            memory_limit_bytes=int(verifier_config["memory_limit_mb"]) * 1024 * 1024,
                            output_limit_bytes=int(verifier_config["output_limit_bytes"]),
                        ).to_dict()
                    record = {
                        "problem_id": problem["problem_id"],
                        "difficulty": problem.get("difficulty"),
                        "sample_index": sample_index,
                        "prompt": prompt,
                        "response": response,
                        "response_length": len(response),
                        "response_tokens": generated.token_count,
                        "finish_reason": generated.finish_reason,
                        "total_tests": len(problem["tests"]),
                        "extraction_success": code is not None,
                        "code": code,
                        "judge": judgement,
                    }
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                completed_ids.add(problem["problem_id"])

    metrics = compute_metrics(records)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the code-generation evaluation pipeline")
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--model-path", help="Override model.name_or_path with a local checkpoint")
    parser.add_argument("--shard-index", type=int, help="Zero-based deterministic problem shard")
    parser.add_argument("--num-shards", type=int, help="Total number of deterministic problem shards")
    parser.add_argument("--resume", help="Resume an existing experiment output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "experiment", "model")
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        if not model_path.is_dir():
            raise ValueError(f"Model checkpoint directory does not exist: {model_path}")
        config["model"]["name_or_path"] = str(model_path)
        config["model"].pop("revision", None)
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together")
    if args.shard_index is not None:
        config = configure_shard(config, args.shard_index, args.num_shards)
    # vLLM must start its worker before any host-side seed helper initializes CUDA.
    # evaluate() seeds the host process before the first generation call.
    generator = create_generator(config)
    output_dir, metrics = evaluate(config, generator, resume=args.resume)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
