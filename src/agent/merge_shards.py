from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.agent.evaluator import configure_agent_shard, load_agent_problems
from src.agent.metrics import compute_agent_metrics
from src.agent.schemas import AgentTrajectory
from src.utils.config import load_config
from src.utils.experiment import collect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and merge Agent evaluation shards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("shards", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        config["model"]["name_or_path"] = str(model_path)
        config["model"].pop("revision", None)
    expected_ids = [problem.problem_id for problem in load_agent_problems(config)]
    trajectories: list[AgentTrajectory] = []
    shard_dirs = [Path(value).resolve() for value in args.shards]
    for index, shard_dir in enumerate(shard_dirs):
        actual = yaml.safe_load((shard_dir / "config.yaml").read_text(encoding="utf-8"))
        if actual != configure_agent_shard(config, index, len(shard_dirs)):
            raise ValueError(f"Agent shard config mismatch: {shard_dir}")
        with (shard_dir / "trajectories.jsonl").open(encoding="utf-8") as handle:
            trajectories.extend(
                AgentTrajectory.from_dict(json.loads(line)) for line in handle if line.strip()
            )
    ids = [trajectory.problem_id for trajectory in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("Agent shards contain duplicate problem IDs")
    if set(ids) != set(expected_ids):
        raise ValueError("Agent shards are incomplete")
    order = {problem_id: index for index, problem_id in enumerate(expected_ids)}
    trajectories.sort(key=lambda trajectory: order[trajectory.problem_id])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories.jsonl").write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in trajectories),
        encoding="utf-8",
    )
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    environment = collect_environment()
    environment["evaluation_shards"] = [str(path) for path in shard_dirs]
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics = compute_agent_metrics(trajectories)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
