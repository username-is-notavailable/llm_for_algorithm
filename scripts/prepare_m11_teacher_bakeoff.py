from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from src.data.agent_eval import read_jsonl, write_jsonl


def stable_key(row: dict, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{row['task_id']}".encode()).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a fixed termination-stratified teacher bake-off")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if not 0 < args.size <= len(rows):
        parser.error("--size must fit the input")
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in sorted(rows, key=lambda value: stable_key(value, args.seed)):
        label = str((row.get("metadata") or {}).get("source_termination_reason") or "unknown")
        buckets[label].append(row)
    labels = sorted(buckets)
    selected = []
    while len(selected) < args.size:
        progressed = False
        for label in labels:
            if buckets[label] and len(selected) < args.size:
                selected.append(buckets[label].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to fill bake-off")
    write_jsonl(args.output, selected)
    output_path = Path(args.output)
    manifest = {
        "schema_version": "m11-teacher-bakeoff-v1",
        "seed": args.seed,
        "input": args.input,
        "size": len(selected),
        "problem_ids": [row["problem_id"] for row in selected],
        "termination_reasons": dict(collections.Counter(
            (row.get("metadata") or {}).get("source_termination_reason") or "unknown"
            for row in selected
        )),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
