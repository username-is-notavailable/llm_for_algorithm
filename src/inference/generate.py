from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedText:
    text: str
    token_count: int
    finish_reason: str | None
    reasoning_content: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class TextGenerator(Protocol):
    def generate_batch(
        self, prompts: list[str], *, num_samples: int, generation: dict[str, Any]
    ) -> list[list[GeneratedText]]: ...


class HuggingFaceGenerator:
    def __init__(self, model_config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_config["name_or_path"]
        revision = model_config.get("revision")
        trust_remote_code = bool(model_config.get("trust_remote_code", False))
        dtype = getattr(torch, model_config.get("dtype", "bfloat16"))
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            device_map=model_config.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        )

    def generate_batch(
        self, prompts: list[str], *, num_samples: int, generation: dict[str, Any]
    ) -> list[list[GeneratedText]]:
        import torch

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        results = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            options = dict(generation)
            options["num_return_sequences"] = num_samples
            if num_samples > 1 and not options.get("do_sample", False):
                raise ValueError("num_samples > 1 requires do_sample=true")
            with torch.inference_mode():
                outputs = self._model.generate(**inputs, **options)
            prompt_tokens = inputs["input_ids"].shape[1]
            generated = outputs[:, prompt_tokens:]
            max_new_tokens = options.get("max_new_tokens")
            results.append([
                GeneratedText(
                    text=self._tokenizer.decode(tokens, skip_special_tokens=True),
                    token_count=int(tokens.shape[0]),
                    finish_reason=(
                        "length"
                        if max_new_tokens is not None and int(tokens.shape[0]) >= int(max_new_tokens)
                        else "stop"
                    ),
                )
                for tokens in generated
            ])
        return results


class VLLMGenerator:
    def __init__(self, model_config: dict[str, Any], inference_config: dict[str, Any]) -> None:
        # CUDA may already have been inspected by the host process. Spawn avoids
        # inheriting an initialized CUDA runtime, which cannot safely be reused
        # by vLLM's EngineCore after fork.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Cloud GPU images commonly provide the CUDA runtime without nvcc.
        # FlashInfer's sampler may JIT-compile during warmup even for greedy
        # decoding, so use vLLM's built-in torch sampler by default.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM

        model_name = model_config["name_or_path"]
        revision = model_config.get("revision")
        self._model = LLM(
            model=model_name,
            revision=revision,
            tokenizer_revision=revision,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            dtype=model_config.get("dtype", "bfloat16"),
            tensor_parallel_size=int(inference_config.get("tensor_parallel_size", 1)),
            max_model_len=int(inference_config.get("max_model_len", 32768)),
            gpu_memory_utilization=float(inference_config.get("gpu_memory_utilization", 0.9)),
            max_num_seqs=int(inference_config.get("max_num_seqs", 8)),
            seed=int(inference_config.get("seed", 0)),
            enforce_eager=bool(inference_config.get("enforce_eager", False)),
        )

    def generate_batch(
        self, prompts: list[str], *, num_samples: int, generation: dict[str, Any]
    ) -> list[list[GeneratedText]]:
        from vllm import SamplingParams

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        options = dict(generation)
        options.pop("do_sample", None)
        if "max_new_tokens" in options:
            options["max_tokens"] = options.pop("max_new_tokens")
        sampling_params = SamplingParams(n=num_samples, **options)
        outputs = self._model.generate(prompts, sampling_params, use_tqdm=True)
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned an unexpected number of requests")
        return [
            [
                GeneratedText(
                    text=candidate.text,
                    token_count=len(candidate.token_ids),
                    finish_reason=candidate.finish_reason,
                )
                for candidate in request.outputs
            ]
            for request in outputs
        ]


def create_generator(config: dict[str, Any]) -> TextGenerator:
    model_config = config["model"]
    backend = model_config.get("backend", "huggingface")
    if backend == "huggingface":
        return HuggingFaceGenerator(model_config)
    if backend == "vllm":
        return VLLMGenerator(model_config, config.get("inference", {}))
    raise ValueError(f"Unsupported inference backend: {backend}")
