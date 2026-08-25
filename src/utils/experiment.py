from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "verl",
            "vllm",
            "flash-attn",
            "openai",
        )
    }
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "packages": packages,
        "cuda": {"available": False},
    }
    try:
        import torch

        available = torch.cuda.is_available()
        info["cuda"] = {
            "available": available,
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count() if available else 0,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ] if available else [],
        }
    except ImportError:
        pass
    return info


def create_experiment_dir(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(experiment["output_dir"]) / f"{experiment['name']}-{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_experiment_metadata(path: Path, config: dict[str, Any], metadata: dict[str, Any]) -> None:
    (path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (path / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
