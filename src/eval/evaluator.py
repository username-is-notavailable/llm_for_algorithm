from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.eval.metrics import compute_metrics
from src.inference.generate import HuggingFaceGenerator, TextGenerator
from src.inference.prompts import build_code_prompt
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(experiment["output_dir"]) / f"{experiment['name']}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def evaluate(config: dict[str, Any], generator: TextGenerator) -> tuple[Path, dict[str, Any]]:
    require_sections(config, "experiment", "model", "prompt", "dataset", "generation", "verifier")
    if config["prompt"].get("template") != "output_protocol_v1":
        raise ValueError("Unsupported prompt template")
    set_seed(int(config["experiment"]["seed"]))
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

    output_dir = _create_output_dir(config)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(collect_environment(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    num_samples = int(config["generation"].get("num_samples", 1))
    generation_options = {
        key: value for key, value in config["generation"].items() if key != "num_samples"
    }
    verifier_config = config["verifier"]
    records: list[dict[str, Any]] = []
    generations_path = output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as output:
        for problem_index, problem in enumerate(problems, start=1):
            print(f"[{problem_index}/{len(problems)}] Evaluating {problem['problem_id']}", flush=True)
            prompt = build_code_prompt(problem["problem"])
            responses = generator.generate(
                prompt,
                num_samples=num_samples,
                generation=generation_options,
            )
            if len(responses) != num_samples:
                raise RuntimeError("Generator returned an unexpected number of samples")
            for sample_index, response in enumerate(responses):
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
                    "total_tests": len(problem["tests"]),
                    "extraction_success": code is not None,
                    "code": code,
                    "judge": judgement,
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

    metrics = compute_metrics(records)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the code-generation evaluation pipeline")
    parser.add_argument("--config", default="configs/eval/default.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "experiment", "model")
    set_seed(int(config["experiment"]["seed"]))
    generator = HuggingFaceGenerator(config["model"])
    output_dir, metrics = evaluate(config, generator)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
