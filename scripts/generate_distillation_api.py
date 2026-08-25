from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.data.agent_eval import read_jsonl, row_test_hash, write_jsonl
from src.data.distill_api import export_queue, run_distillation_workers
from src.data.repair_queue import RepairQueue
from src.utils.config import load_config, require_sections
from src.utils.experiment import collect_environment


def prepare_output(config: dict[str, Any], resume: str | None) -> Path:
    if resume:
        output = Path(resume)
        saved = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Resume config does not match saved distillation config")
        return output
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
    return output


def add_tasks(queue: RepairQueue, rows: list[dict[str, Any]], stage: str) -> None:
    for row in rows:
        queue.add(f"{stage}:{row['problem_id']}", row)


def rejection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(collections.Counter(row.get("rejection_reason") or "unknown" for row in rows))


def load_frozen_rows(input_config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = Path(input_config["dataset"])
    manifest = json.loads(Path(input_config["manifest"]).read_text(encoding="utf-8"))
    split = input_config.get("manifest_split", "train")
    rows = read_jsonl(dataset_path)
    if [row["problem_id"] for row in rows] != manifest["problem_ids"][split]:
        raise ValueError(f"Distillation dataset does not match frozen manifest split {split}")
    actual_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_sha256 != manifest["dataset_sha256"]:
        raise ValueError("Distillation dataset SHA-256 does not match frozen manifest")
    for row in rows:
        if row_test_hash(row) != manifest["test_sha256"][row["problem_id"]]:
            raise ValueError(f"Distillation test split hash mismatch: {row['problem_id']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate verifier-gated 8B/32B distillation data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(
        config,
        "experiment",
        "input",
        "api",
        "teachers",
        "generation",
        "acceptance",
        "verifier",
        "queue",
    )
    output = prepare_output(config, args.resume)
    rows = load_frozen_rows(config["input"])

    primary = config["teachers"]["primary"]
    primary_queue = RepairQueue(output / "primary_tasks.sqlite3")
    add_tasks(primary_queue, rows, primary["stage"])
    primary_queue.reclaim_stale(int(config["queue"].get("lease_timeout_seconds", 3600)))
    primary_counts = run_distillation_workers(
        primary_queue, config, primary, escalated=False
    )
    export_queue(primary_queue, output, "primary")

    primary_rejected = primary_queue.export("rejected")
    escalation = config["teachers"]["escalation"]
    escalation_queue = RepairQueue(output / "escalation_tasks.sqlite3")
    add_tasks(escalation_queue, [row["problem"] for row in primary_rejected], escalation["stage"])
    escalation_queue.reclaim_stale(int(config["queue"].get("lease_timeout_seconds", 3600)))
    escalation_counts = run_distillation_workers(
        escalation_queue, config, escalation, escalated=True
    )
    export_queue(escalation_queue, output, "escalation")

    accepted = primary_queue.export("accepted") + escalation_queue.export("accepted")
    accepted.sort(key=lambda row: row["problem_id"])
    write_jsonl(output / "accepted.jsonl", accepted)
    escalation_rejected = escalation_queue.export("rejected")
    metrics = {
        "input_problems": len(rows),
        "primary": {
            "model": primary["model"],
            "queue_status": primary_counts,
            "rejection_reasons": rejection_counts(primary_rejected),
        },
        "escalation": {
            "model": escalation["model"],
            "input_problems": len(primary_rejected),
            "queue_status": escalation_counts,
            "rejection_reasons": rejection_counts(escalation_rejected),
        },
        "accepted_total": len(accepted),
        "unresolved_total": len(rows) - len(accepted),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
