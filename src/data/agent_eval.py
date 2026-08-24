from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


def _stable_digest(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def select_agent_dev(
    rows: Iterable[dict[str, Any]], *, smoke_ids: list[str], size: int, seed: int
) -> list[dict[str, Any]]:
    values = list(rows)
    by_id = {row["problem_id"]: row for row in values}
    if len(by_id) != len(values):
        raise ValueError("Agent source contains duplicate problem IDs")
    missing = [problem_id for problem_id in smoke_ids if problem_id not in by_id]
    if missing:
        raise ValueError(f"Agent source is missing smoke IDs: {missing}")
    if not len(smoke_ids) <= size <= len(values):
        raise ValueError("Agent dev size must include smoke and fit the source")

    smoke_set = set(smoke_ids)
    buckets: dict[str, deque[dict[str, Any]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        if row["problem_id"] not in smoke_set:
            grouped[str(row.get("difficulty") or "unknown")].append(row)
    for label, bucket in grouped.items():
        bucket.sort(key=lambda row: _stable_digest(row["problem_id"], seed))
        buckets[label] = deque(bucket)

    selected = [by_id[problem_id] for problem_id in smoke_ids]
    labels = sorted(buckets)
    while len(selected) < size:
        progressed = False
        for label in labels:
            if buckets[label] and len(selected) < size:
                selected.append(buckets[label].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to fill Agent dev split")
    return selected


def split_visible_hidden_tests(
    row: dict[str, Any], *, seed: int, visible_fraction: float, visible_max: int
) -> dict[str, Any]:
    tests = row.get("tests")
    if not isinstance(tests, list) or len(tests) < 2:
        raise ValueError(f"{row.get('problem_id')}: at least two tests are required")
    if not 0 < visible_fraction < 1 or visible_max < 1:
        raise ValueError("Invalid visible test selection")
    order = sorted(
        range(len(tests)),
        key=lambda index: _stable_digest(f"{row['problem_id']}:{index}", seed),
    )
    visible_count = min(visible_max, max(1, round(len(tests) * visible_fraction)))
    visible_count = min(visible_count, len(tests) - 1)
    visible_indices = sorted(order[:visible_count])
    visible_set = set(visible_indices)
    hidden_indices = [index for index in range(len(tests)) if index not in visible_set]
    value = dict(row)
    value["visible_tests"] = [tests[index] for index in visible_indices]
    value["hidden_tests"] = [tests[index] for index in hidden_indices]
    # Existing one-shot evaluator consumes `tests`; those are exactly the same
    # hidden tests used by Agent final evaluation.
    value["tests"] = value["hidden_tests"]
    value["agent_test_split"] = {
        "visible_indices": visible_indices,
        "hidden_indices": hidden_indices,
    }
    return value


def row_test_hash(row: dict[str, Any]) -> str:
    payload = {
        "problem_id": row["problem_id"],
        "visible_tests": row["visible_tests"],
        "hidden_tests": row["hidden_tests"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
