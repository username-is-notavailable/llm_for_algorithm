from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_HF_HOME = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(PROJECT_HF_HOME))
os.environ.setdefault("HF_XET_CACHE", str(PROJECT_HF_HOME / "xet"))

from huggingface_hub import snapshot_download


MODELS = {
    "4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned official Qwen3 models")
    parser.add_argument("sizes", nargs="+", choices=MODELS, help="Model sizes to cache")
    args = parser.parse_args()

    cache_dir = PROJECT_HF_HOME / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for size in dict.fromkeys(args.sizes):
        repo_id, revision = MODELS[size]
        print(f"Downloading {repo_id}@{revision} to {cache_dir}", flush=True)
        path = snapshot_download(repo_id, revision=revision, cache_dir=cache_dir)
        print(f"Cached {size}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
