from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.livecodebench import adapt_rows, split_manifest, stratified_splits, write_jsonl
from src.utils.config import load_config, require_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen LiveCodeBench v1 splits")
    parser.add_argument("--config", default="configs/data/livecodebench_v1.yaml")
    parser.add_argument("--source-jsonl", help="Use an existing source JSONL instead of downloading")
    return parser.parse_args()


def _source_paths(config: dict, override: str | None) -> list[Path]:
    if override:
        return [Path(override)]
    project_cache = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
    if project_cache.is_dir():
        os.environ.setdefault("HF_HOME", str(project_cache))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to download LiveCodeBench") from error
    source = config["source"]
    return [
        Path(hf_hub_download(
            repo_id=source["dataset"],
            filename=filename,
            revision=source["revision"],
            repo_type="dataset",
        ))
        for filename in source["filenames"]
    ]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "source", "split", "output")
    paths = _source_paths(config, args.source_jsonl)
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    accepted, rejected = adapt_rows(rows)
    split_config = config["split"]
    splits = stratified_splits(
        accepted,
        seed=int(split_config["seed"]),
        dev_fraction=float(split_config["dev_fraction"]),
        smoke_size=int(split_config["smoke_size"]),
        max_problems=int(split_config["max_problems"]),
    )
    output_dir = Path(config["output"]["directory"])
    write_jsonl(output_dir / "eval_v1.jsonl", splits["eval"])
    write_jsonl(output_dir / "dev_v1.jsonl", splits["dev"])
    write_jsonl(output_dir / "smoke_10.jsonl", splits["smoke"])
    (output_dir / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    source = config["source"]
    manifest = split_manifest(
        splits,
        dataset=source["dataset"],
        release=source["release"],
        revision=source["revision"],
        seed=int(split_config["seed"]),
        selection={
            "method": "difficulty_stratified_fixed_seed",
            "dev_fraction": float(split_config["dev_fraction"]),
            "smoke_size": int(split_config["smoke_size"]),
            "max_problems": int(split_config["max_problems"]),
        },
    )
    manifest["source_counts"] = {
        "rows": len(rows),
        "accepted_stdin": len(accepted),
        "rejected": len(rejected),
        "selected": len(splits["eval"]) + len(splits["dev"]),
    }
    manifest_path = Path(config["output"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), **manifest["counts"]}, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
