from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from src.data.agent_eval import read_jsonl
from src.data.repair_api import export_results, run_workers
from src.data.repair_queue import RepairQueue
from src.utils.config import load_config, require_sections
from src.utils.experiment import collect_environment


def prepare_run(config: dict, resume: str | None) -> tuple[Path, RepairQueue]:
    if resume:
        output = Path(resume)
        saved = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Resume config does not match saved repair config")
    else:
        experiment = config["experiment"]
        output = Path(experiment["output_dir"]) / (
            f"{experiment['name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (output / "environment.json").write_text(
            json.dumps(collect_environment(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    queue = RepairQueue(output / "tasks.sqlite3")
    for row in read_jsonl(config["input"]["failure_pool"]):
        queue.add(row["task_id"], row)
    queue.reclaim_stale(int(config["queue"].get("lease_timeout_seconds", 3600)))
    return output, queue


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate verified repair trajectories via API")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "experiment", "input", "api", "generation", "agent", "verifier", "queue")
    output, queue = prepare_run(config, args.resume)
    counts = run_workers(queue, config)
    export_results(queue, output)
    (output / "metrics.json").write_text(
        json.dumps({"queue_status": counts}, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
