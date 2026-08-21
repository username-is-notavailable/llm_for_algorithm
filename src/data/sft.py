from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict, deque
from typing import Any, Iterable

from src.data.leakage import normalized_problem, problem_sha256, problem_simhash
from src.verifier import extract_code

_THINK_RE = re.compile(r"<think(?:\s[^>]*)?>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)


def normalize_difficulty(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text.isdigit():
        rating = int(text)
        if rating <= 5:
            return "easy" if rating <= 2 else "medium" if rating == 3 else "hard"
        return "easy" if rating <= 1200 else "medium" if rating <= 1900 else "hard"
    if text in {"easy", "basic", "beginner", "1", "2"}:
        return "easy"
    if text in {"medium", "medium_hard", "interview", "3"}:
        return "medium"
    if text in {"hard", "very_hard", "extreme", "advanced", "4", "5"}:
        return "hard"
    return "unknown"


def split_generation(response: str) -> tuple[str, str] | None:
    if not isinstance(response, str):
        return None
    matches = _THINK_RE.findall(response)
    reasoning = max((match.strip() for match in matches), key=len, default="")
    code = extract_code(response)
    if not reasoning or not code:
        return None
    return reasoning, code.strip()


def render_response(reasoning: str, code: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n```cpp\n{code.strip()}\n```"


def adapt_ocr2(row: dict[str, Any], question: str) -> dict[str, Any]:
    parsed = split_generation(row.get("r1_generation", ""))
    if parsed is None:
        raise ValueError("unparseable_generation")
    reasoning, code = parsed
    if not question or question.strip() == "-":
        raise ValueError("missing_question")
    raw_pass_rate = str(row.get("pass_rate", "-1")).strip()
    try:
        pass_rate = float(raw_pass_rate)
    except ValueError as error:
        raise ValueError("invalid_pass_rate") from error
    return {
        "problem_id": f"ocr2:{row['question_id']}",
        "sample_id": f"ocr2:{row['id']}",
        "source": "nvidia/OpenCodeReasoning-2",
        "problem": question.strip(),
        "difficulty": normalize_difficulty(row.get("difficulty")),
        "tags": [],
        "reasoning": reasoning,
        "code": code,
        "response": render_response(reasoning, code),
        "language": "cpp",
        "verified": pass_rate >= 0,
        "metadata": {
            "original_id": row["id"],
            "original_question_id": row["question_id"],
            "dataset": row.get("dataset"),
            "dataset_split": row.get("split"),
            "dataset_index": row.get("index"),
            "platform": row.get("source"),
            "difficulty_raw": row.get("difficulty"),
            "judgement": row.get("judgement"),
            "pass_rate": pass_rate,
            "license": row.get("license"),
        },
    }


def quality_reason(row: dict[str, Any], config: dict[str, Any]) -> str | None:
    if str(row.get("judgement", "")).lower() != str(config["require_judgement"]).lower():
        return "judgement"
    try:
        pass_rate = float(row.get("pass_rate", -1))
    except (TypeError, ValueError):
        return "pass_rate"
    if pass_rate < 0 and not config["allow_unverified"]:
        return "unverified"
    if 0 <= pass_rate < float(config["minimum_pass_rate"]):
        return "pass_rate"
    parsed = split_generation(row.get("r1_generation", ""))
    if parsed is None:
        return "generation"
    reasoning, code = parsed
    solution = row.get("solution")
    if isinstance(solution, str) and solution.strip():
        if "".join(code.split()) != "".join(solution.split()):
            return "solution_mismatch"
    if len(reasoning) < int(config["minimum_reasoning_characters"]):
        return "reasoning_too_short"
    if len(row["r1_generation"]) > int(config["maximum_response_characters"]):
        return "response_too_long"
    if not code:
        return "code"
    return None


def deduplicate_candidates(rows: Iterable[dict[str, Any]], filter_config: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    best: dict[str, dict[str, Any]] = {}
    rejected: Counter = Counter()
    for row in rows:
        reason = quality_reason(row, filter_config)
        if reason:
            rejected[reason] += 1
            continue
        question_id = str(row.get("question_id", ""))
        if not question_id:
            rejected["question_id"] += 1
            continue
        current = best.get(question_id)
        score = (float(row.get("pass_rate", -1)), -len(row["r1_generation"]), str(row.get("id", "")))
        if current is None:
            best[question_id] = row
        else:
            current_score = (float(current.get("pass_rate", -1)), -len(current["r1_generation"]), str(current.get("id", "")))
            if score > current_score:
                best[question_id] = row
            rejected["duplicate_problem"] += 1
    return list(best.values()), rejected


def is_eval_leak(problem: str, fingerprints: list[dict[str, Any]], max_hamming_distance: int = 6) -> str | None:
    digest = problem_sha256(problem)
    fingerprint = problem_simhash(problem)
    length = len(normalized_problem(problem))
    for item in fingerprints:
        if digest == item["sha256"]:
            return "eval_exact"
        if min(length, int(item["normalized_length"])) >= 100 and (fingerprint ^ int(item["simhash"])).bit_count() <= max_hamming_distance:
            return "eval_near_duplicate"
    return None


def stable_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['problem_id']}".encode()).digest())


def balanced_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in stable_order(rows, seed):
        key = (row["difficulty"], str(row["metadata"].get("platform", "unknown")))
        groups[key].append(row)
    result: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys:
        next_keys = []
        for key in keys:
            result.append(groups[key].popleft())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return result
