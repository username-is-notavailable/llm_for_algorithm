from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, require_sections
from src.utils.experiment import collect_environment, create_experiment_dir, save_experiment_metadata
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 Milestone 0 GPU smoke test")
    parser.add_argument("--config", default="configs/environment/smoke.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "experiment", "model", "generation", "environment")
    output_dir = create_experiment_dir(config)
    logger = configure_logging(output_dir / "smoke_test.log")
    set_seed(int(config["experiment"]["seed"]))

    environment = collect_environment()
    save_experiment_metadata(output_dir, config, environment)
    logger.info("Environment: %s", environment)

    required = config["environment"]
    if required.get("require_cuda") and not environment["cuda"]["available"]:
        raise RuntimeError("CUDA is required but unavailable")
    for package in ("verl", "vllm"):
        if required.get(f"require_{package}") and environment["packages"][package] is None:
            raise RuntimeError(f"Required package is not installed: {package}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    dtype_name = model_config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"], trust_remote_code=model_config.get("trust_remote_code", False)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        dtype=dtype,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    inputs = tokenizer(config["generation"]["prompt"], return_tensors="pt").to(model.device)
    generation_config = {k: v for k, v in config["generation"].items() if k != "prompt"}
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_config)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    (output_dir / "generation.txt").write_text(generated, encoding="utf-8")
    logger.info("Generated text:\n%s", generated)
    logger.info("CUDA allocated bytes: %d", torch.cuda.memory_allocated())
    logger.info("CUDA peak allocated bytes: %d", torch.cuda.max_memory_allocated())
    logger.info("Smoke test passed; artifacts: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
