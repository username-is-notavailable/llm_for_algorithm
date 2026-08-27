from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from typing import Any, Callable

from src.inference.generate import GeneratedText


class SlidingWindowLimiter:
    """Shared request/token reservation limiter for concurrent API workers."""

    def __init__(self, *, requests_per_minute: int, tokens_per_minute: int) -> None:
        if requests_per_minute < 1 or tokens_per_minute < 1:
            raise ValueError("API rate limits must be positive")
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, reserved_tokens: int) -> None:
        if not 0 < reserved_tokens <= self._tpm:
            raise ValueError("reserved_tokens must fit within tokens_per_minute")
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= 60:
                    self._events.popleft()
                used = sum(tokens for _, tokens in self._events)
                if len(self._events) < self._rpm and used + reserved_tokens <= self._tpm:
                    self._events.append((now, reserved_tokens))
                    return
                wait = max(0.05, 60 - (now - self._events[0][0]))
            time.sleep(min(wait, 1.0))


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
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


class DashScopeAgentGenerator:
    """Streaming DashScope generator through its OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        limiter: SlidingWindowLimiter | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = dict(config)
        self._model = str(config["model"])
        self._limiter = limiter
        self._sleep = sleep
        if client is None:
            api_key_env = str(config.get("api_key_env", "DASHSCOPE_API_KEY"))
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=str(
                    config.get(
                        "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                ),
                timeout=float(config.get("timeout_seconds", 180)),
                max_retries=0,
            )
        self._client = client

    def generate(self, messages: list[dict[str, str]], generation: dict[str, Any]) -> GeneratedText:
        max_tokens = int(generation.get("max_new_tokens", self._config.get("max_tokens", 8192)))
        thinking_budget = self._config.get("thinking_budget")
        if thinking_budget is not None and int(thinking_budget) < 1:
            raise ValueError("thinking_budget must be a positive integer")
        reserved = (
            max_tokens
            + int(self._config.get("prompt_token_reserve", 8192))
            + (int(thinking_budget) if thinking_budget is not None else 0)
        )
        attempts = int(self._config.get("max_retries", 5)) + 1
        for attempt in range(attempts):
            try:
                if self._limiter:
                    self._limiter.acquire(reserved)
                return self._request(messages, generation, max_tokens)
            except Exception:
                if attempt + 1 == attempts:
                    raise
                delay = min(
                    float(self._config.get("retry_max_seconds", 30)),
                    float(self._config.get("retry_base_seconds", 1)) * (2**attempt),
                )
                self._sleep(delay * (0.8 + random.random() * 0.4))
        raise AssertionError("unreachable")

    def _request(
        self, messages: list[dict[str, str]], generation: dict[str, Any], max_tokens: int
    ) -> GeneratedText:
        extra_body: dict[str, Any] = {
            "enable_thinking": bool(self._config.get("enable_thinking", True))
        }
        if self._config.get("thinking_budget") is not None:
            extra_body["thinking_budget"] = int(self._config["thinking_budget"])
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": normalize_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        for source, target in (("temperature", "temperature"), ("top_p", "top_p")):
            if source in generation:
                kwargs[target] = generation[source]
        reasoning: list[str] = []
        content: list[str] = []
        finish_reason = None
        request_id = None
        completion_tokens = 0
        prompt_tokens = None
        for chunk in self._client.chat.completions.create(**kwargs):
            request_id = request_id or getattr(chunk, "id", None)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = choice.delta
            value = getattr(delta, "reasoning_content", None)
            if value:
                reasoning.append(value)
            value = getattr(delta, "content", None)
            if value:
                content.append(value)
        text = "".join(content)
        reported_completion_tokens = completion_tokens or None
        effective_completion_tokens = completion_tokens or max(1, len(text) // 4)
        return GeneratedText(
            text=text,
            token_count=effective_completion_tokens,
            finish_reason=finish_reason,
            reasoning_content="".join(reasoning) or None,
            provider_metadata={
                "provider": "dashscope",
                "model": self._model,
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": reported_completion_tokens,
                "completion_tokens_estimated": completion_tokens == 0,
            },
        )
