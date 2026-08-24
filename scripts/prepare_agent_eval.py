from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.agent_eval import (
    read_jsonl,
    row_test_hash,
    select_agent_dev,
    split_visible_hidden_tests,
    write_jsonl,
)
from src.utils.config import load_config, require_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen visible/hidden Agent eval splits")
    parser.add_argument("--config", default="configs/data/agent_eval_v1.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "selection", "output")
    source = config["source"]
    frozen = json.loads(Path(source["manifest"]).read_text(encoding="utf-8"))
    smoke_ids = list(frozen["problem_ids"]["smoke"])
    dev_ids = list(frozen["problem_ids"]["dev"])
    rows = read_jsonl(source["path"])
    by_id = {row["problem_id"]: row for row in rows}
    if list(by_id) != dev_ids:
        raise ValueError("Source dev rows do not match the frozen M3 manifest order")

    selection = config["selection"]
    selected = select_agent_dev(
        rows,
        smoke_ids=smoke_ids,
        size=int(selection["dev_size"]),
        seed=int(selection["seed"]),
    )
    adapted = [
        split_visible_hidden_tests(
            row,
            seed=int(selection["seed"]),
            visible_fraction=float(selection["visible_fraction"]),
            visible_max=int(selection["visible_max"]),
        )
        for row in selected
    ]
    smoke = adapted[: len(smoke_ids)]
    output = config["output"]
    write_jsonl(output["smoke"], smoke)
    write_jsonl(output["dev"], adapted)
    manifest = {
        "schema_version": "agent-eval-v1",
        "source_manifest": source["manifest"],
        "selection": selection,
        "problem_ids": {
            "smoke": [row["problem_id"] for row in smoke],
            "dev": [row["problem_id"] for row in adapted],
        },
        "test_sha256": {row["problem_id"]: row_test_hash(row) for row in adapted},
        "counts": {
            "smoke": len(smoke),
            "dev": len(adapted),
            "visible_tests": sum(len(row["visible_tests"]) for row in adapted),
            "hidden_tests": sum(len(row["hidden_tests"]) for row in adapted),
        },
    }
    manifest_path = Path(output["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
