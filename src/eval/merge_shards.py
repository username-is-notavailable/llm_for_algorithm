from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.eval.evaluator import configure_shard, load_problems, validate_split_manifest
from src.eval.metrics import compute_metrics
from src.utils.config import load_config
from src.utils.experiment import collect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and merge deterministic eval shards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("shards", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        config["model"]["name_or_path"] = str(model_path)
        config["model"].pop("revision", None)

    problems = load_problems(config["dataset"]["path"])
    manifest = config["dataset"].get("manifest")
    if manifest:
        validate_split_manifest(problems, manifest, config["dataset"]["manifest_split"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        problems = problems[: int(limit)]
    expected_ids = [row["problem_id"] for row in problems]
    expected_samples = int(config["generation"].get("num_samples", 1))

    shard_dirs = [Path(value).resolve() for value in args.shards]
    records: list[dict] = []
    for index, shard_dir in enumerate(shard_dirs):
        expected_config = configure_shard(config, index, len(shard_dirs))
        actual_config = yaml.safe_load((shard_dir / "config.yaml").read_text(encoding="utf-8"))
        if actual_config != expected_config:
            raise ValueError(f"Shard config mismatch: {shard_dir}")
        with (shard_dir / "generations.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())

    keys = [(row["problem_id"], int(row["sample_index"])) for row in records]
    expected_keys = [(problem_id, index) for problem_id in expected_ids for index in range(expected_samples)]
    if len(keys) != len(set(keys)):
        raise ValueError("Merged shards contain duplicate problem/sample records")
    if set(keys) != set(expected_keys):
        missing = set(expected_keys) - set(keys)
        extra = set(keys) - set(expected_keys)
        raise ValueError(f"Merged shards are incomplete: missing={len(missing)}, extra={len(extra)}")

    order = {key: index for index, key in enumerate(expected_keys)}
    records.sort(key=lambda row: order[(row["problem_id"], int(row["sample_index"]))])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "generations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    environment = collect_environment()
    environment["evaluation_shards"] = [str(path) for path in shard_dirs]
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics = compute_metrics(records)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
