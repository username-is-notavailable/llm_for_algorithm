from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalized_problem(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def problem_sha256(text: str) -> str:
    return hashlib.sha256(normalized_problem(text).encode()).hexdigest()


def problem_simhash(text: str) -> int:
    tokens = normalized_problem(text).split()
    features = tokens if len(tokens) < 3 else [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def find_leaks(
    eval_rows: Iterable[dict[str, Any]], training_rows: Iterable[dict[str, Any]], *, max_hamming_distance: int = 6
) -> list[dict[str, Any]]:
    eval_index = []
    for row in eval_rows:
        text = row["problem"]
        eval_index.append((row["problem_id"], problem_sha256(text), problem_simhash(text), len(normalized_problem(text))))
    leaks: list[dict[str, Any]] = []
    for train in training_rows:
        train_id = train["problem_id"]
        text = train["problem"]
        digest = problem_sha256(text)
        fingerprint = problem_simhash(text)
        normalized_length = len(normalized_problem(text))
        for eval_id, eval_digest, eval_fingerprint, eval_length in eval_index:
            reason = None
            if train_id == eval_id:
                reason = "problem_id"
            elif digest == eval_digest:
                reason = "normalized_sha256"
            elif min(normalized_length, eval_length) >= 100 and (fingerprint ^ eval_fingerprint).bit_count() <= max_hamming_distance:
                reason = "near_duplicate_simhash"
            if reason:
                leaks.append({"training_problem_id": train_id, "eval_problem_id": eval_id, "reason": reason})
                break
    return leaks


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
