from __future__ import annotations

import base64
import hashlib
import json
import pickletools
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DIFFICULTIES = ("easy", "medium", "hard")


def _json_value(value: Any, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {field}") from error
    return value


def decode_private_tests(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, str):
        decoded = value
    else:
        try:
            payload = zlib.decompress(base64.b64decode(value))
            try:
                serialized = payload.decode("utf-8")
            except UnicodeDecodeError:
                allowed_ops = {"PROTO", "FRAME", "BINUNICODE", "SHORT_BINUNICODE", "MEMOIZE", "STOP"}
                operations = list(pickletools.genops(payload))
                if any(operation.name not in allowed_ops for operation, _, _ in operations):
                    raise ValueError("Unsafe pickle operation in private_test_cases")
                strings = [argument for operation, argument, _ in operations if operation.name in {"BINUNICODE", "SHORT_BINUNICODE"}]
                if len(strings) != 1 or not isinstance(strings[0], str):
                    raise ValueError("Unexpected pickle structure in private_test_cases")
                serialized = strings[0]
            decoded = json.loads(serialized)
        except (ValueError, zlib.error, UnicodeDecodeError) as error:
            raise ValueError("Invalid encoded private_test_cases") from error
    if not isinstance(decoded, list):
        raise ValueError("private_test_cases must decode to a list")
    return decoded


def _standard_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    public = _json_value(row.get("public_test_cases", []), "public_test_cases")
    private = decode_private_tests(row.get("private_test_cases"))
    if not isinstance(public, list):
        raise ValueError("public_test_cases must be a list")
    tests: list[dict[str, str]] = []
    for case in [*public, *private]:
        if not isinstance(case, dict) or case.get("testtype", "stdin") != "stdin":
            raise ValueError("Only stdin test cases are supported")
        stdin, stdout = case.get("input"), case.get("output")
        if not isinstance(stdin, str) or not isinstance(stdout, str):
            raise ValueError("Test input and output must be strings")
        tests.append({"input": stdin, "output": stdout})
    if not tests:
        raise ValueError("Problem has no tests")
    return tests


def adapt_problem(row: dict[str, Any]) -> dict[str, Any]:
    required = ("question_id", "question_content", "platform", "contest_date", "difficulty")
    missing = [field for field in required if not isinstance(row.get(field), str) or not row[field]]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    difficulty = row["difficulty"].lower()
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    platform = row["platform"].lower()
    source_id = row["question_id"]
    metadata = _json_value(row.get("metadata", {}), "metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    return {
        "problem_id": f"livecodebench:{platform}:{source_id}",
        "source": "livecodebench/code_generation_lite",
        "problem": row["question_content"],
        "language": "cpp",
        "difficulty": difficulty,
        "tests": _standard_tests(row),
        "metadata": {
            "question_id": source_id,
            "question_title": row.get("question_title", ""),
            "platform": platform,
            "contest_id": row.get("contest_id", ""),
            "contest_date": row["contest_date"],
            "starter_code": row.get("starter_code", ""),
            **metadata,
        },
    }


def adapt_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        try:
            problem = adapt_problem(row)
            if problem["problem_id"] in seen:
                raise ValueError("Duplicate problem_id")
            seen.add(problem["problem_id"])
            accepted.append(problem)
        except ValueError as error:
            rejected.append({"row": str(index), "question_id": str(row.get("question_id", "")), "reason": str(error)})
    return accepted, rejected


def _stable_shuffle(problems: list[dict[str, Any]], seed: int, namespace: str) -> list[dict[str, Any]]:
    result = list(problems)
    random.Random(f"{seed}:{namespace}").shuffle(result)
    return result


def stratified_splits(
    problems: list[dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float = 0.2,
    smoke_size: int = 10,
    max_problems: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in problems:
        groups[problem["difficulty"]].append(problem)
    if set(groups) != set(DIFFICULTIES):
        raise ValueError("All three difficulty levels are required")
    if max_problems is not None and len(problems) > max_problems:
        if max_problems < len(DIFFICULTIES):
            raise ValueError("max_problems is too small for stratification")
        exact = {difficulty: max_problems * len(groups[difficulty]) / len(problems) for difficulty in DIFFICULTIES}
        quotas = {difficulty: int(exact[difficulty]) for difficulty in DIFFICULTIES}
        remaining = max_problems - sum(quotas.values())
        for difficulty in sorted(DIFFICULTIES, key=lambda value: (exact[value] - quotas[value], value), reverse=True)[:remaining]:
            quotas[difficulty] += 1
        groups = {
            difficulty: _stable_shuffle(groups[difficulty], seed, f"candidate:{difficulty}")[: quotas[difficulty]]
            for difficulty in DIFFICULTIES
        }
    dev: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        shuffled = _stable_shuffle(groups[difficulty], seed, difficulty)
        dev_count = max(1, round(len(shuffled) * dev_fraction))
        dev.extend(shuffled[:dev_count])
        evaluation.extend(shuffled[dev_count:])
    dev = _stable_shuffle(dev, seed, "dev")
    evaluation = _stable_shuffle(evaluation, seed, "eval")
    if len(dev) < smoke_size:
        raise ValueError("Development split is smaller than smoke_size")
    smoke = _stable_shuffle(dev, seed, "smoke")[:smoke_size]
    return {"dev": dev, "eval": evaluation, "smoke": smoke}


def split_manifest(
    splits: dict[str, list[dict[str, Any]]],
    *,
    dataset: str,
    release: str,
    revision: str,
    seed: int,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = {name: [problem["problem_id"] for problem in rows] for name, rows in splits.items()}
    payload = json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "dataset": dataset,
        "release": release,
        "revision": revision,
        "seed": seed,
        "selection": selection or {},
        "split_sha256": hashlib.sha256(payload).hexdigest(),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "difficulty_counts": {
            name: {difficulty: sum(row["difficulty"] == difficulty for row in rows) for difficulty in DIFFICULTIES}
            for name, rows in splits.items()
        },
        "problem_ids": ids,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
