from __future__ import annotations


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
