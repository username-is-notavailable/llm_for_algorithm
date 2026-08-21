from __future__ import annotations

from typing import Any, Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, num_samples: int, generation: dict[str, Any]) -> list[str]: ...


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

    def generate(self, prompt: str, *, num_samples: int, generation: dict[str, Any]) -> list[str]:
        import torch

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        options = dict(generation)
        options["num_return_sequences"] = num_samples
        if num_samples > 1 and not options.get("do_sample", False):
            raise ValueError("num_samples > 1 requires do_sample=true")
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **options)
        prompt_tokens = inputs["input_ids"].shape[1]
        return [
            self._tokenizer.decode(output[prompt_tokens:], skip_special_tokens=True)
            for output in outputs
        ]
