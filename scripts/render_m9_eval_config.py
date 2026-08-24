from __future__ import annotations

import argparse
from pathlib import Path

import yaml


MODELS = {
    "1.7b": (
        "Qwen/Qwen3-1.7B-Base",
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "qwen3-1.7b-base",
    ),
    "1.7b-post": (
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "qwen3-1.7b-posttrained",
    ),
    "4b": (
        "Qwen/Qwen3-4B-Base",
        "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "qwen3-4b-base",
    ),
    "4b-post": (
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        "qwen3-4b-posttrained",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a pinned M9 model-size config")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-size", choices=MODELS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-total-generation-tokens", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.input).read_text(encoding="utf-8"))
    name, revision, experiment_slug = MODELS[args.model_size]
    config["model"]["name_or_path"] = name
    config["model"]["revision"] = revision
    config["experiment"]["name"] = config["experiment"]["name"].replace(
        "qwen3-1.7b-base", experiment_slug
    )
    if args.max_new_tokens is not None:
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive")
        config["generation"]["max_new_tokens"] = args.max_new_tokens
        config["experiment"]["name"] += f"-long{args.max_new_tokens // 1024}k"
    if args.max_total_generation_tokens is not None:
        if "agent" not in config:
            raise ValueError("--max-total-generation-tokens only applies to Agent configs")
        config["agent"]["max_total_generation_tokens"] = args.max_total_generation_tokens
    Path(args.output).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
