from typing import Any

from src.agent.generate import ChatAgentGenerator
from src.inference.generate import GeneratedText


class FakeTokenizer:
    def __init__(self, token_count: int = 10) -> None:
        self.token_count = token_count
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **options: Any) -> str:
        assert options["add_generation_prompt"] is True
        self.messages = messages
        return "rendered prompt"

    def encode(self, prompt: str) -> list[int]:
        assert prompt == "rendered prompt"
        return list(range(self.token_count))


class FakeTextGenerator:
    def __init__(self) -> None:
        self.generation: dict[str, Any] = {}

    def generate_batch(
        self, prompts: list[str], *, num_samples: int, generation: dict[str, Any]
    ) -> list[list[GeneratedText]]:
        assert prompts == ["rendered prompt"]
        assert num_samples == 1
        self.generation = generation
        return [[GeneratedText("code", 1, "stop")]]


def test_chat_agent_maps_tool_observation_to_user_and_clamps_context() -> None:
    tokenizer = FakeTokenizer(token_count=90)
    text = FakeTextGenerator()
    generator = ChatAgentGenerator(tokenizer, text, max_model_len=100)
    result = generator.generate(
        [
            {"role": "assistant", "content": "candidate"},
            {"role": "tool", "content": "wrong answer"},
        ],
        {"max_new_tokens": 50},
    )
    assert result.text == "code"
    assert tokenizer.messages[-1]["role"] == "user"
    assert "Execution environment observation" in tokenizer.messages[-1]["content"]
    assert text.generation["max_new_tokens"] == 10


def test_chat_agent_stops_before_overflowing_context() -> None:
    generator = ChatAgentGenerator(FakeTokenizer(token_count=100), FakeTextGenerator(), max_model_len=100)
    result = generator.generate([], {"max_new_tokens": 10})
    assert result.finish_reason == "length"
    assert result.token_count == 0
