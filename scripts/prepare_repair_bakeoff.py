from __future__ import annotations

import argparse
import collections
import hashlib
import json

from src.data.agent_eval import read_jsonl, write_jsonl


def select(rows: list[dict], *, size: int, seed: int) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['task_id']}".encode()).digest(),
    )
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in ordered:
        buckets[str(row["problem"].get("difficulty") or "unknown")].append(row)
    selected: list[dict] = []
    while len(selected) < size and any(buckets.values()):
        for difficulty in ("easy", "medium", "hard", "unknown"):
            if buckets[difficulty] and len(selected) < size:
                selected.append(buckets[difficulty].pop(0))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a deterministic, difficulty-balanced repair bake-off")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    selected = select(rows, size=min(args.size, len(rows)), seed=args.seed)
    write_jsonl(args.output, selected)
    print(
        json.dumps(
            {
                "input_tasks": len(rows),
                "selected_tasks": len(selected),
                "difficulty": dict(
                    collections.Counter(
                        row["problem"].get("difficulty") or "unknown" for row in selected
                    )
                ),
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
