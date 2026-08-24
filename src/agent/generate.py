from __future__ import annotations

from typing import Any

from src.inference.generate import GeneratedText, TextGenerator, create_generator


class ChatAgentGenerator:
    """Render Agent messages and delegate generation to an existing text backend."""

    def __init__(
        self,
        tokenizer: Any,
        text_generator: TextGenerator,
        *,
        enable_thinking: bool | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._text_generator = text_generator
        self._enable_thinking = enable_thinking
        self._max_model_len = max_model_len

    def _render(self, messages: list[dict[str, str]]) -> str:
        # We deliberately do not use native function-calling messages. The v1
        # environment has two textual actions, and execution feedback is a
        # controller-owned observation represented as the next user turn.
        normalized = [
            {
                "role": "user" if message["role"] == "tool" else message["role"],
                "content": (
                    "Execution environment observation:\n" + message["content"]
                    if message["role"] == "tool"
                    else message["content"]
                ),
            }
            for message in messages
        ]
        options: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking is not None:
            options["enable_thinking"] = self._enable_thinking
        return self._tokenizer.apply_chat_template(normalized, **options)

    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        prompt = self._render(messages)
        options = dict(generation)
        if self._max_model_len is not None and "max_new_tokens" in options:
            prompt_tokens = len(self._tokenizer.encode(prompt))
            remaining = self._max_model_len - prompt_tokens
            if remaining <= 0:
                return GeneratedText(text="", token_count=0, finish_reason="length")
            options["max_new_tokens"] = min(int(options["max_new_tokens"]), remaining)
        result = self._text_generator.generate_batch([prompt], num_samples=1, generation=options)
        return result[0][0]


def create_agent_generator(config: dict[str, Any]) -> ChatAgentGenerator:
    from transformers import AutoTokenizer

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["name_or_path"],
        revision=model.get("revision"),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
    )
    return ChatAgentGenerator(
        tokenizer,
        create_generator(config),
        enable_thinking=config.get("prompt", {}).get("enable_thinking"),
        max_model_len=int(config.get("inference", {}).get("max_model_len", 32768)),
    )
