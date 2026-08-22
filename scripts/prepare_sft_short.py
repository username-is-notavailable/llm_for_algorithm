from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SFT_10K_SHA256 = "16d25b5ad5780b4b5925a6a504210c11c7d39f35b535e7c783e6f3e9398a3581"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a bounded-response SFT subset")
    parser.add_argument("--input", default="data/processed/sft_10k.jsonl")
    parser.add_argument("--output", default="data/processed/sft_1k_short_v1.jsonl")
    parser.add_argument("--stats", default="data/processed/sft_1k_short_v1_stats.json")
    parser.add_argument("--response-max-tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--expected-input-sha256", default=SFT_10K_SHA256)
    return parser.parse_args()


def select_short_rows(rows: list[dict], *, response_max_tokens: int, samples: int) -> list[dict]:
    if response_max_tokens < 1 or samples < 1:
        raise ValueError("response_max_tokens and samples must be positive")
    eligible = [
        row
        for row in rows
        if bool(row.get("verified"))
        and int(row["token_counts"]["response"]) <= response_max_tokens
    ]
    selected = eligible[:samples]
    if len(selected) != samples:
        raise ValueError(f"Only {len(selected)} eligible rows; need {samples}")
    problem_ids = [row["problem_id"] for row in selected]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("Selected rows contain duplicate problem IDs")
    return selected


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    input_payload = input_path.read_bytes()
    input_sha256 = hashlib.sha256(input_payload).hexdigest()
    if input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Input SHA-256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    rows = [json.loads(line) for line in input_payload.decode().splitlines() if line.strip()]
    selected = select_short_rows(
        rows, response_max_tokens=args.response_max_tokens, samples=args.samples
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    output_path.write_text(payload, encoding="utf-8")
    response_lengths = [int(row["token_counts"]["response"]) for row in selected]
    total_lengths = [int(row["token_counts"]["total"]) for row in selected]
    stats = {
        "source": str(input_path),
        "source_sha256": input_sha256,
        "output": str(output_path),
        "samples": len(selected),
        "unique_problems": len({row["problem_id"] for row in selected}),
        "response_max_tokens": args.response_max_tokens,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "response_tokens": {
            "p50": percentile(response_lengths, 0.50),
            "p90": percentile(response_lengths, 0.90),
            "p95": percentile(response_lengths, 0.95),
            "p99": percentile(response_lengths, 0.99),
            "max": max(response_lengths),
            "mean": sum(response_lengths) / len(response_lengths),
        },
        "total_tokens": {
            "p50": percentile(total_lengths, 0.50),
            "p90": percentile(total_lengths, 0.90),
            "max": max(total_lengths),
        },
        "difficulty": dict(collections.Counter(str(row.get("difficulty") or "unknown") for row in selected)),
        "platform": dict(collections.Counter(str(row.get("metadata", {}).get("platform") or "unknown") for row in selected)),
    }
    Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
