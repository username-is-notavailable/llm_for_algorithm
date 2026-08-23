from __future__ import annotations

from typing import Any, Callable


def build_code_prompt(problem: str) -> str:
    """Render a competitive-programming problem using Output Protocol v1."""

    return f"""You are solving a competitive programming problem.
Reason step by step, then provide one complete GNU C++17 program.

Your response must use exactly this structure:
<think>
Your reasoning, proof of correctness, and complexity analysis.
</think>

```cpp
Your complete C++17 program.
```

Problem:
{problem.strip()}
"""


def build_code_only_prompt(problem: str) -> str:
    """Render a concise code-only request for instruction-tuned coder models."""

    return f"""Solve the following competitive programming problem.
Return one complete GNU C++17 program in a ```cpp code fence. Do not omit any code.

Problem:
{problem.strip()}
"""


def create_prompt_builder(
    prompt_config: dict[str, Any], model_config: dict[str, Any]
) -> Callable[[str], str]:
    template = prompt_config.get("template")
    if template == "output_protocol_v1":
        return build_code_prompt
    if template not in {"qwen3_chat_output_protocol_v1", "chat_code_only_v1"}:
        raise ValueError(f"Unsupported prompt template: {template}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )

    def render(problem: str) -> str:
        content = (
            build_code_only_prompt(problem)
            if template == "chat_code_only_v1"
            else build_code_prompt(problem)
        )
        options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if template == "qwen3_chat_output_protocol_v1":
            options["enable_thinking"] = bool(prompt_config.get("enable_thinking", True))
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            **options,
        )

    return render
