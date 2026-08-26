from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.data.agent_eval import read_jsonl, write_jsonl
from src.training.sft import encode_agent_sft_row


def trajectory_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    steps = row["repair_trajectory"]["steps"]
    if not steps:
        raise ValueError(f"{row['task_id']}: trajectory has no steps")
    final = steps[-1]
    if final["submission"]["effective_action"] != "final":
        raise ValueError(f"{row['task_id']}: trajectory does not end in final")
    messages = [dict(message) for message in final["prompt_messages"]]
    messages.append({"role": "assistant", "content": final["submission"]["response"]})
    if any(message["role"] == "tool" and "Private" in message["content"] for message in messages):
        # Revealed counterexamples are allowed, but unrevealed private results must never be serialized.
        if "One failing case has now been revealed" not in "\n".join(
            message["content"] for message in messages if message["role"] == "tool"
        ):
            raise ValueError(f"{row['task_id']}: possible private-test leakage")
    return messages


def stable_key(problem_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare multi-turn Agent SFT smoke data")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/codecontests_plus_repair_v1/repair_api_8b_canonical_33.jsonl",
            "data/processed/codecontests_plus_repair_v1/repair_api_32b_canonical_9.jsonl",
        ],
    )
    parser.add_argument("--train-output", default="data/processed/agent_sft_v1/train_34.jsonl")
    parser.add_argument("--dev-output", default="data/processed/agent_sft_v1/dev_8.jsonl")
    parser.add_argument("--manifest", default="data/splits/agent_sft_smoke_v1_manifest.json")
    parser.add_argument("--dev-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    project_cache = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(project_cache))

    source_rows = [row for path in args.inputs for row in read_jsonl(path)]
    if len({row["problem_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Agent SFT source contains duplicate problem IDs")
    rows = []
    for source in source_rows:
        rows.append(
            {
                "schema_version": "agent-sft-messages-v1",
                "problem_id": source["problem_id"],
                "task_id": source["task_id"],
                "source": source["source"],
                "teacher_model": source["teacher_model"],
                "messages": trajectory_messages(source),
                "metadata": {
                    "trajectory_schema": source["repair_trajectory"]["schema_version"],
                    "normalization": source.get("normalization"),
                },
            }
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    for row in rows:
        encoded = encode_agent_sft_row(row, tokenizer, args.max_length)
        row["token_counts"] = {
            "total": encoded["length"],
            "assistant": sum(label != -100 for label in encoded["labels"]),
            "context": sum(label == -100 for label in encoded["labels"]),
        }
    ordered = sorted(rows, key=lambda row: stable_key(row["problem_id"], args.seed))
    if not 0 < args.dev_size < len(ordered):
        raise ValueError("dev-size must leave non-empty train and dev splits")
    dev, train = ordered[: args.dev_size], ordered[args.dev_size :]
    write_jsonl(args.train_output, train)
    write_jsonl(args.dev_output, dev)

    def digest(path: str) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "schema_version": "agent-sft-smoke-v1",
        "source_files": args.inputs,
        "tokenizer": {"name_or_path": args.tokenizer, "revision": args.revision},
        "max_length": args.max_length,
        "seed": args.seed,
        "problem_ids": {
            "train": [row["problem_id"] for row in train],
            "dev": [row["problem_id"] for row in dev],
        },
        "counts": {
            "train": len(train),
            "dev": len(dev),
            "tokens": {
                split: {
                    key: sum(row["token_counts"][key] for row in values)
                    for key in ("total", "assistant", "context")
                }
                for split, values in (("train", train), ("dev", dev))
            },
        },
        "sha256": {"train": digest(args.train_output), "dev": digest(args.dev_output)},
    }
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"Manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
