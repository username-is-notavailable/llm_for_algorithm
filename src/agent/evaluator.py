from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.agent import AgentConfig, AgentProblem, ExecutionLimits, LocalVerifierBackend
from src.agent.controller import run_agent
from src.agent.generate import create_agent_generator
from src.agent.metrics import compute_agent_metrics
from src.agent.schemas import AgentTrajectory
from src.data.agent_eval import read_jsonl, row_test_hash
from src.utils.config import load_config, require_sections
from src.utils.experiment import collect_environment
from src.utils.reproducibility import set_seed
from src.verifier import TestCase


def configure_agent_shard(config: dict[str, Any], index: int, count: int) -> dict[str, Any]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid Agent shard index/count")
    value = copy.deepcopy(config)
    value["experiment"]["name"] += f"-shard-{index + 1:02d}-of-{count:02d}"
    value["dataset"]["shard"] = {"index": index, "count": count}
    return value


def load_agent_problems(config: dict[str, Any]) -> list[AgentProblem]:
    dataset = config["dataset"]
    rows = read_jsonl(dataset["path"])
    manifest = json.loads(Path(dataset["manifest"]).read_text(encoding="utf-8"))
    split = dataset["manifest_split"]
    expected_ids = manifest["problem_ids"][split]
    if [row["problem_id"] for row in rows] != expected_ids:
        raise ValueError(f"Agent dataset does not match frozen manifest split {split}")
    for row in rows:
        if row_test_hash(row) != manifest["test_sha256"][row["problem_id"]]:
            raise ValueError(f"Agent test split hash mismatch: {row['problem_id']}")
    limit = dataset.get("limit")
    if limit is not None:
        rows = rows[: int(limit)]
    shard = dataset.get("shard")
    if shard:
        rows = rows[int(shard["index"]) :: int(shard["count"])]
    verifier = config["verifier"]
    limits = ExecutionLimits(
        compile_timeout_seconds=float(verifier["compile_timeout_seconds"]),
        execution_timeout_seconds=float(verifier["execution_timeout_seconds"]),
        memory_limit_bytes=int(verifier["memory_limit_mb"]) * 1024 * 1024,
        output_limit_bytes=int(verifier["output_limit_bytes"]),
    )
    return [
        AgentProblem(
            problem_id=row["problem_id"],
            problem=row["problem"],
            visible_tests=tuple(TestCase(**test) for test in row["visible_tests"]),
            hidden_tests=tuple(TestCase(**test) for test in row["hidden_tests"]),
            difficulty=row.get("difficulty"),
            source=row.get("source"),
            limits=limits,
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def _output_dir(config: dict[str, Any], resume: str | Path | None) -> Path:
    if resume:
        output = Path(resume)
        if not output.is_dir():
            raise ValueError(f"Resume directory does not exist: {output}")
        saved = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Resume config does not match saved Agent config")
        return output
    experiment = config["experiment"]
    timestamp = os.environ.get("AGENT_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(experiment["output_dir"]) / f"{experiment['name']}-{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(collect_environment(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def _load_existing(path: Path, known_ids: set[str]) -> tuple[list[AgentTrajectory], set[str]]:
    if not path.is_file():
        return [], set()
    trajectories = []
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            trajectory = AgentTrajectory.from_dict(json.loads(line))
            if trajectory.problem_id not in known_ids:
                raise ValueError(f"Resume line {line_number}: unknown problem ID")
            if trajectory.problem_id in completed:
                raise ValueError(f"Resume line {line_number}: duplicate problem ID")
            trajectories.append(trajectory)
            completed.add(trajectory.problem_id)
    return trajectories, completed


def evaluate_agent(
    config: dict[str, Any], generator: Any, *, resume: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    require_sections(
        config, "experiment", "model", "prompt", "dataset", "generation", "verifier", "agent"
    )
    set_seed(int(config["experiment"]["seed"]))
    problems = load_agent_problems(config)
    output = _output_dir(config, resume)
    path = output / "trajectories.jsonl"
    trajectories, completed = _load_existing(path, {problem.problem_id for problem in problems})
    if completed:
        print(f"Resuming with {len(completed)}/{len(problems)} trajectories complete", flush=True)
    agent_config = AgentConfig(**config["agent"])
    generation = {key: value for key, value in config["generation"].items() if key != "num_samples"}
    if int(config["generation"].get("num_samples", 1)) != 1:
        raise ValueError("Agent Evaluation v1 requires one trajectory per problem")
    backend = LocalVerifierBackend()
    with path.open("a", encoding="utf-8") as handle:
        for index, problem in enumerate(problems, 1):
            if problem.problem_id in completed:
                continue
            print(f"[{index}/{len(problems)}] Agent rollout {problem.problem_id}", flush=True)
            trajectory = run_agent(
                trajectory_id=f"{config['experiment']['name']}:{problem.problem_id}",
                problem=problem,
                model=config["model"],
                config=agent_config,
                generator=generator,
                backend=backend,
                generation=generation,
            )
            trajectories.append(trajectory)
            handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(problem.problem_id)
    metrics = compute_agent_metrics(trajectories)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded execution-guided Agent evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        if not model_path.is_dir():
            raise ValueError(f"Model path does not exist: {model_path}")
        config["model"]["name_or_path"] = str(model_path)
        config["model"].pop("revision", None)
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together")
    if args.shard_index is not None:
        config = configure_agent_shard(config, args.shard_index, args.num_shards)
    generator = create_agent_generator(config)
    output, metrics = evaluate_agent(config, generator, resume=args.resume)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
