from __future__ import annotations

import argparse
from pathlib import Path

import yaml


MODELS = {
    "1.7b": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
    "4b": ("Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a pinned M9 model-size config")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-size", choices=MODELS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.input).read_text(encoding="utf-8"))
    name, revision = MODELS[args.model_size]
    config["model"]["name_or_path"] = name
    config["model"]["revision"] = revision
    config["experiment"]["name"] = config["experiment"]["name"].replace(
        "qwen3-1.7b", f"qwen3-{args.model_size}"
    )
    Path(args.output).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
