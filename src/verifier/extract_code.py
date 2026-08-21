from __future__ import annotations

import re
from dataclasses import dataclass


_FENCE_RE = re.compile(
    r"```[ \t]*(?P<language>[^\n`]*)\n(?P<code>.*?)```",
    flags=re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer(?:\s[^>]*)?>(.*?)</answer\s*>", flags=re.DOTALL | re.IGNORECASE)
_CPP_LANGUAGES = {"cpp", "c++", "cc", "cxx"}


@dataclass(frozen=True)
class _Candidate:
    priority: int
    code: str


def _looks_like_cpp(text: str) -> bool:
    markers = ("#include", "using namespace std", "std::")
    main_function = re.search(r"\b(?:int|signed|auto)\s+main\s*\(", text)
    return main_function is not None or any(marker in text for marker in markers)


def _fenced_candidates(response: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for match in _FENCE_RE.finditer(response):
        language_parts = match.group("language").strip().lower().split(maxsplit=1)
        language = language_parts[0] if language_parts else ""
        code = match.group("code").strip()
        if not code:
            continue
        if language in _CPP_LANGUAGES:
            priority = 2
        elif not language:
            priority = 1
        else:
            priority = 0
        candidates.append(_Candidate(priority=priority, code=code))
    return candidates


def extract_code(response: str) -> str | None:
    """Extract the most plausible C++ program from a model response."""

    if not isinstance(response, str) or not response.strip():
        return None

    candidates = _fenced_candidates(response)
    if candidates:
        cpp_candidates = [candidate for candidate in candidates if candidate.priority == 2]
        if cpp_candidates:
            return max(cpp_candidates, key=lambda candidate: len(candidate.code)).code
        plain_candidates = [candidate for candidate in candidates if candidate.priority == 1]
        plausible_plain = [candidate for candidate in plain_candidates if _looks_like_cpp(candidate.code)]
        if plausible_plain:
            return max(plausible_plain, key=lambda candidate: len(candidate.code)).code
        return None

    answer_candidates = [candidate.strip() for candidate in _ANSWER_RE.findall(response)]
    plausible_answers = [candidate for candidate in answer_candidates if _looks_like_cpp(candidate)]
    if plausible_answers:
        return max(plausible_answers, key=len)

    # Salvage a malformed response with an opening fence but no closing fence.
    opening = re.search(r"```[ \t]*(?:cpp|c\+\+|cc|cxx)?[^\n`]*\n", response, flags=re.IGNORECASE)
    if opening:
        remainder = response[opening.end() :].strip()
        if _looks_like_cpp(remainder):
            return remainder

    stripped = response.strip()
    return stripped if _looks_like_cpp(stripped) else None
