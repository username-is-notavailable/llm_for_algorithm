from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.leakage import normalized_problem, problem_sha256, problem_simhash, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/processed/livecodebench_v1/eval_v1.jsonl")
    parser.add_argument("--output", default="data/splits/eval_v1_fingerprints.json")
    args = parser.parse_args()
    rows = read_jsonl(args.eval)
    fingerprints = [
        {
            "problem_id": row["problem_id"],
            "sha256": problem_sha256(row["problem"]),
            "simhash": problem_simhash(row["problem"]),
            "normalized_length": len(normalized_problem(row["problem"])),
        }
        for row in rows
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version": 1, "fingerprints": fingerprints}, indent=2), encoding="utf-8")
    print(f"Wrote {len(fingerprints)} fingerprints to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
