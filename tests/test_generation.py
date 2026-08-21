from __future__ import annotations

import sys
from types import SimpleNamespace

from src.inference.generate import VLLMGenerator, create_generator


def test_vllm_generator_batches_prompts_and_translates_options(monkeypatch) -> None:
    captured = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            captured["options"] = kwargs

    class FakeModel:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params
            captured["use_tqdm"] = use_tqdm
            return [
                SimpleNamespace(outputs=[SimpleNamespace(
                    text=f"{prompt}:{index}", token_ids=[1, 2], finish_reason="stop"
                ) for index in range(2)])
                for prompt in prompts
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=FakeSamplingParams))
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._model = FakeModel()

    responses = generator.generate_batch(
        ["a", "b"],
        num_samples=2,
        generation={"max_new_tokens": 128, "temperature": 0.0, "do_sample": False, "seed": 42},
    )

    assert [[value.text for value in row] for row in responses] == [["a:0", "a:1"], ["b:0", "b:1"]]
    assert all(value.token_count == 2 and value.finish_reason == "stop" for row in responses for value in row)
    assert captured["prompts"] == ["a", "b"]
    assert captured["options"] == {"n": 2, "max_tokens": 128, "temperature": 0.0, "seed": 42}
    assert captured["use_tqdm"] is True


def test_generator_factory_rejects_unknown_backend() -> None:
    try:
        create_generator({"model": {"backend": "unknown"}})
    except ValueError as error:
        assert "Unsupported inference backend" in str(error)
    else:
        raise AssertionError("Expected an unsupported backend error")
