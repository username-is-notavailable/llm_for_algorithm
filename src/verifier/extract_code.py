from __future__ import annotations

import re
from dataclasses import dataclass


_FENCE_RE = re.compile(
    r"```[ \t]*(?P<language>[^\n`]*)\n(?P<code>.*?)```",
    flags=re.DOTALL,
)
_THINK_RE = re.compile(r"<think(?:\s[^>]*)?>.*?</think\s*>", flags=re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"</?answer(?:\s[^>]*)?>", flags=re.IGNORECASE)
_FILE_HEADER_RE = re.compile(
    r"<file(?:\s[^>]*)?>[^<\n]*</file>\s*", flags=re.IGNORECASE
)
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


def _select_fenced_candidate(candidates: list[_Candidate]) -> str | None:
    cpp_candidates = [candidate for candidate in candidates if candidate.priority == 2]
    if cpp_candidates:
        return max(cpp_candidates, key=lambda candidate: len(candidate.code)).code
    plain_candidates = [candidate for candidate in candidates if candidate.priority == 1]
    plausible_plain = [candidate for candidate in plain_candidates if _looks_like_cpp(candidate.code)]
    if plausible_plain:
        return max(plausible_plain, key=lambda candidate: len(candidate.code)).code
    return None


def _normalize_raw_candidate(response: str) -> str:
    """Remove filename wrappers emitted around otherwise raw source code."""

    stripped = _FILE_HEADER_RE.sub("", response).strip()
    # Some coding models emit a closing Markdown fence after a <file> header
    # without emitting the corresponding opening fence.
    return re.sub(r"\n[ \t]*```[ \t]*$", "", stripped).strip()


def extract_code(response: str) -> str | None:
    """Extract the most plausible C++ program from a model response."""

    if not isinstance(response, str) or not response.strip():
        return None

    response_without_thinking = _THINK_RE.sub("", response)
    normalized_response = _ANSWER_TAG_RE.sub("", response_without_thinking)
    candidates = _fenced_candidates(normalized_response)
    if candidates:
        return _select_fenced_candidate(candidates)

    # Salvage a malformed response with an opening fence but no closing fence.
    opening = re.search(
        r"```[ \t]*(?:cpp|c\+\+|cc|cxx)?[^\n`]*\n",
        normalized_response,
        flags=re.IGNORECASE,
    )
    if opening:
        remainder = normalized_response[opening.end() :].strip()
        if _looks_like_cpp(remainder):
            return remainder

    stripped = _normalize_raw_candidate(normalized_response)
    return stripped if _looks_like_cpp(stripped) else None
