import pytest

from src.verifier.extract_code import extract_code


@pytest.mark.parametrize("language", ["cpp", "c++", ""])
def test_extracts_supported_code_fences(language: str) -> None:
    response = f"Explanation\n```{language}\n#include <iostream>\nint main() {{}}\n```"
    assert extract_code(response) == "#include <iostream>\nint main() {}"


def test_prefers_longest_cpp_block_among_multiple_blocks() -> None:
    response = """
```python
print('not C++')
```
```cpp
int main() {}
```
```c++
#include <iostream>
int main() { std::cout << 1; }
```
"""
    assert extract_code(response) == "#include <iostream>\nint main() { std::cout << 1; }"


def test_extracts_answer_wrapper_and_malformed_fence() -> None:
    assert extract_code("<answer>#include <iostream>\nint main() {}</answer>") == (
        "#include <iostream>\nint main() {}"
    )
    assert extract_code("```cpp\n#include <iostream>\nint main() {}") == (
        "#include <iostream>\nint main() {}"
    )


def test_rejects_response_without_cpp_code() -> None:
    assert extract_code("The solution uses dynamic programming.") is None
    assert extract_code("```python\nprint(1)\n```") is None


def test_extracts_raw_cpp_without_code_fence() -> None:
    response = "signed main() { return 0; }"
    assert extract_code(response) == response
