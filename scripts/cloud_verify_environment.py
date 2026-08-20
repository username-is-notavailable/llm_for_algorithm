from __future__ import annotations

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


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
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

    project_root = Path(__file__).resolve().parents[1]
    verl_root = Path(os.environ.get("VERL_ROOT", project_root / ".third_party" / "verl"))
    commit = subprocess.run(
        ["git", "-C", str(verl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_VERL_COMMIT:
        raise RuntimeError(f"Unexpected verl commit: {commit}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    tensor = torch.ones(1, device="cuda")
    if tensor.item() != 1:
        raise RuntimeError("CUDA tensor smoke test failed")

    driver_version = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "python": sys.version.split()[0],
        "verl_commit": commit,
        "packages": versions,
        "nvidia_driver": driver_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
