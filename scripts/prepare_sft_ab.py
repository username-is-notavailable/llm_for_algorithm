from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.data.sft import balanced_order
from src.inference.prompts import build_code_only_prompt


SFT_10K_SHA256 = "16d25b5ad5780b4b5925a6a504210c11c7d39f35b535e7c783e6f3e9398a3581"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive matched short-reasoning and code-only SFT datasets"
    )
    parser.add_argument("--input", default="data/processed/sft_10k.jsonl")
    parser.add_argument(
        "--short-output", default="data/processed/sft_1k_short_reasoning_v2.jsonl"
    )
    parser.add_argument("--code-output", default="data/processed/sft_1k_code_only_v2.jsonl")
    parser.add_argument("--manifest", default="data/processed/sft_1k_ab_v2_manifest.json")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2048)
    parser.add_argument("--response-max-tokens", type=int, default=4096)
    parser.add_argument("--total-max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--expected-input-sha256", default=SFT_10K_SHA256)
    return parser.parse_args()


def select_matched_rows(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    reasoning_max_tokens: int,
    response_max_tokens: int,
    total_max_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    limits = (samples, reasoning_max_tokens, response_max_tokens, total_max_tokens)
    if any(value < 1 for value in limits):
        raise ValueError("samples and token limits must be positive")
    eligible = [
        row
        for row in rows
        if bool(row.get("verified"))
        and isinstance(row.get("reasoning"), str)
        and bool(row["reasoning"].strip())
        and isinstance(row.get("code"), str)
        and bool(row["code"].strip())
        and int(row["token_counts"]["reasoning"]) <= reasoning_max_tokens
        and int(row["token_counts"]["response"]) <= response_max_tokens
        and int(row["token_counts"]["total"]) <= total_max_tokens
    ]
    selected = balanced_order(eligible, seed)[:samples]
    if len(selected) != samples:
        raise ValueError(f"Only {len(eligible)} eligible rows; need {samples}")
    problem_ids = [str(row["problem_id"]) for row in selected]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("Selected rows contain duplicate problem IDs")
    return selected, len(eligible)


def render_code_only_response(code: str) -> str:
    return f"```cpp\n{code.strip()}\n```"


def derive_variants(
    rows: list[dict[str, Any]], token_count: Callable[[str], int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    short_rows: list[dict[str, Any]] = []
    code_rows: list[dict[str, Any]] = []
    for source in rows:
        short = dict(source)
        short["metadata"] = dict(source.get("metadata") or {})
        short["metadata"]["sft_variant"] = "short_reasoning_v2"
        short_rows.append(short)

        code_only = dict(source)
        code_only["reasoning"] = ""
        code_only["prompt"] = build_code_only_prompt(str(source["problem"]))
        code_only["response"] = render_code_only_response(str(source["code"]))
        code_only["metadata"] = dict(source.get("metadata") or {})
        code_only["metadata"]["sft_variant"] = "code_only_v2"
        code_only["token_counts"] = dict(source["token_counts"])
        code_only["token_counts"].update(
            {
                "prompt": token_count(code_only["prompt"]),
                "reasoning": 0,
                "code": token_count(str(code_only["code"])),
                "response": token_count(code_only["response"]),
            }
        )
        code_only["token_counts"]["total"] = (
            code_only["token_counts"]["prompt"] + code_only["token_counts"]["response"]
        )
        code_rows.append(code_only)
    return short_rows, code_rows


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def dataset_stats(rows: list[dict[str, Any]], payload: str) -> dict[str, Any]:
    response_tokens = [int(row["token_counts"]["response"]) for row in rows]
    reasoning_tokens = [int(row["token_counts"]["reasoning"]) for row in rows]
    total_tokens = [int(row["token_counts"]["total"]) for row in rows]

    def summary(values: list[int]) -> dict[str, int | float]:
        return {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "samples": len(rows),
        "unique_problems": len({row["problem_id"] for row in rows}),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "reasoning_tokens": summary(reasoning_tokens),
        "response_tokens": summary(response_tokens),
        "total_tokens": summary(total_tokens),
        "difficulty": dict(
            collections.Counter(str(row.get("difficulty") or "unknown") for row in rows)
        ),
        "platform": dict(
            collections.Counter(
                str((row.get("metadata") or {}).get("platform") or "unknown") for row in rows
            )
        ),
    }


def serialize(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    input_payload = input_path.read_bytes()
    input_sha256 = hashlib.sha256(input_payload).hexdigest()
    if input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Input SHA-256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    source_rows = [
        json.loads(line) for line in input_payload.decode().splitlines() if line.strip()
    ]
    selected, eligible_count = select_matched_rows(
        source_rows,
        samples=args.samples,
        reasoning_max_tokens=args.reasoning_max_tokens,
        response_max_tokens=args.response_max_tokens,
        total_max_tokens=args.total_max_tokens,
        seed=args.seed,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        trust_remote_code=False,
    )
    short_rows, code_rows = derive_variants(
        selected, lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )
    short_payload = serialize(short_rows)
    code_payload = serialize(code_rows)
    short_path = Path(args.short_output)
    code_path = Path(args.code_output)
    short_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    short_path.write_text(short_payload, encoding="utf-8")
    code_path.write_text(code_payload, encoding="utf-8")

    manifest = {
        "version": "sft_ab_v2",
        "source": str(input_path),
        "source_sha256": input_sha256,
        "selection": {
            "seed": args.seed,
            "samples": args.samples,
            "eligible_rows": eligible_count,
            "reasoning_max_tokens": args.reasoning_max_tokens,
            "response_max_tokens": args.response_max_tokens,
            "total_max_tokens": args.total_max_tokens,
            "problem_ids": [row["problem_id"] for row in selected],
        },
        "tokenizer": {"name_or_path": TOKENIZER_NAME, "revision": TOKENIZER_REVISION},
        "variants": {
            "short_reasoning": {
                "path": str(short_path),
                **dataset_stats(short_rows, short_payload),
            },
            "code_only": {
                "path": str(code_path),
                **dataset_stats(code_rows, code_payload),
            },
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
