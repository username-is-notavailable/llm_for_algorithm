from __future__ import annotations

from types import SimpleNamespace

from src.inference.dashscope import DashScopeAgentGenerator, normalize_messages


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        yield SimpleNamespace(
            id="request-1",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(reasoning_content="inspect error", content=None),
                )
            ],
        )
        yield SimpleNamespace(
            id="request-1",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(reasoning_content=None, content="<action>final</action>"),
                )
            ],
        )
        yield SimpleNamespace(
            id="request-1",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[],
        )


def test_dashscope_stream_keeps_reasoning_and_answer_separate() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = DashScopeAgentGenerator(
        {"model": "qwen3-8b", "enable_thinking": True, "max_retries": 0}, client=client
    )
    result = generator.generate(
        [{"role": "tool", "content": "compile failed"}], {"max_new_tokens": 100}
    )
    assert result.text == "<action>final</action>"
    assert result.reasoning_content == "inspect error"
    assert result.token_count == 20
    assert result.provider_metadata["request_id"] == "request-1"
    assert completions.kwargs["extra_body"] == {"enable_thinking": True}
    assert completions.kwargs["messages"][0]["role"] == "user"


def test_normalize_messages_marks_execution_observation() -> None:
    assert normalize_messages([{"role": "tool", "content": "WA"}]) == [
        {"role": "user", "content": "Execution environment observation:\nWA"}
    ]


def test_dashscope_passes_thinking_budget() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = DashScopeAgentGenerator(
        {
            "model": "qwen3-235b-a22b-thinking-2507",
            "enable_thinking": True,
            "thinking_budget": 16384,
            "max_retries": 0,
        },
        client=client,
    )
    generator.generate([{"role": "user", "content": "solve"}], {"max_new_tokens": 1024})
    assert completions.kwargs["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 16384,
    }
