from __future__ import annotations

import argparse
import json

from src.data.leakage import find_leaks, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject SFT/GRPO data overlapping fixed evaluation data")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--training", required=True, nargs="+")
    parser.add_argument("--report", default="outputs/data/leakage_report.json")
    args = parser.parse_args()
    eval_rows = read_jsonl(args.eval)
    leaks = []
    for path in args.training:
        leaks.extend({"training_file": path, **leak} for leak in find_leaks(eval_rows, read_jsonl(path)))
    from pathlib import Path

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(leaks, indent=2), encoding="utf-8")
    print(f"Leakage matches: {len(leaks)}; report: {report}")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
