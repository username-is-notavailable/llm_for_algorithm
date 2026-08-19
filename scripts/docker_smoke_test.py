from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_VERL_COMMIT = "b256ebf83b304d83be5c1207fdf6867c04a0d077"
EXPECTED_VERSIONS = {
    "torch": "2.11.0",
    "transformers": "5.5.3",
    "vllm": "0.24.0",
    "flash-attn": "2.8.3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the pinned verl training container")
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Only validate imports and versions (used while building the image)",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    args = parse_args()

    versions = {name: package_version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if not versions[name].startswith(expected)
    }
    if mismatches:
        raise RuntimeError(f"Pinned dependency mismatch: {mismatches}")

    import flash_attn
    import ray
    import tensordict
    import torch
    import transformers
    import verl
    import vllm

    del flash_attn, ray, tensordict, transformers, verl, vllm

    verl_root = Path(os.environ.get("VERL_ROOT", "/opt/verl"))
    commit = subprocess.run(
        ["git", "-C", str(verl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_VERL_COMMIT:
        raise RuntimeError(f"Unexpected verl commit: {commit}")

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "verl_commit": commit,
        "packages": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if not args.skip_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
        report["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        tensor = torch.ones(1, device="cuda")
        if tensor.item() != 1:
            raise RuntimeError("CUDA tensor smoke test failed")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
