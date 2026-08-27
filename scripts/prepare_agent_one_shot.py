from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.agent.backend import LocalVerifierBackend, full_gate_problem
from src.agent.controller import build_initial_messages
from src.agent.schemas import AgentConfig
from src.data.agent_eval import write_jsonl
from src.data.problem_store import IndexedProblemStore
from src.data.repair_api import load_problem
from src.utils.config import load_config, require_sections


def stream_jsonl(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build verified one-shot Agent SFT messages")
    parser.add_argument("--config", default="configs/data/m11_agent_one_shot_300_v1.yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "input", "output", "selection", "agent", "verifier")
    if args.max_samples is not None:
        if args.max_samples < 1:
            parser.error("--max-samples must be positive")
        config["selection"]["max_samples"] = args.max_samples
    if args.output:
        config["output"]["dataset"] = args.output
    if args.manifest:
        config["output"]["manifest"] = args.manifest
    input_config = config["input"]
    store = IndexedProblemStore(input_config["problem_dataset"], input_config["problem_index"])
    backend = LocalVerifierBackend()
    agent_config = AgentConfig(
        max_execute_calls=int(config["agent"]["max_execute_calls"]),
        max_candidate_submissions=int(config["agent"]["max_execute_calls"]) + 1,
        max_feedback_bytes=int(config["agent"]["max_feedback_bytes"]),
    )
    rows = []
    for position, seed in enumerate(stream_jsonl(input_config["seeds"]), 1):
        if len(rows) >= int(config["selection"]["max_samples"]):
            break
        problem_row = store.get(seed["problem_id"])
        problem = load_problem(problem_row, config["verifier"])
        code = seed["code"].strip()
        visible = backend.execute_visible(
            code,
            problem,
            executions_remaining=agent_config.max_execute_calls - 1,
            max_feedback_bytes=agent_config.max_feedback_bytes,
        )
        full = backend.evaluate_hidden(code, full_gate_problem(problem))
        if visible.visible_pass_rate != 1 or not full.success:
            raise RuntimeError(f"Frozen one-shot seed no longer passes: {seed['problem_id']}")
        messages = build_initial_messages(problem, agent_config)
        response = f"```cpp\n{code}\n```"
        messages.extend([
            {
                "role": "assistant",
                "content": f"<action>execute_code</action>\n{response}",
                "trainable": True,
            },
            {"role": "tool", "content": visible.model_feedback},
            {
                "role": "assistant",
                "content": f"<action>final</action>\n{response}",
                "trainable": True,
            },
        ])
        rows.append({
            "schema_version": "agent-sft-messages-v4",
            "problem_id": problem.problem_id,
            "task_id": f"one-shot:{problem.problem_id}",
            "source": problem.source,
            "teacher_model": "CodeContests+ verified correct submission",
            "messages": messages,
            "metadata": {
                "trajectory_type": "one_shot_execute_pass_final",
                "source_judge": seed.get("source_judge"),
            },
        })
        if position % 25 == 0:
            print(f"Verified one-shot {len(rows)}/{config['selection']['max_samples']}", flush=True)

    output = config["output"]
    write_jsonl(output["dataset"], rows)
    manifest = {
        "schema_version": "agent-one-shot-300-v1",
        "config": config,
        "counts": {"samples": len(rows), "assistant_targets": len(rows) * 2},
        "problem_ids": [row["problem_id"] for row in rows],
        "sha256": sha256_file(output["dataset"]),
    }
    manifest_path = Path(output["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
