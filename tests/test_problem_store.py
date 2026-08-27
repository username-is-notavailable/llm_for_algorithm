import json

import pytest

from src.data.problem_store import IndexedProblemStore


def test_indexed_problem_store_reads_records_by_byte_offset(tmp_path) -> None:
    dataset = tmp_path / "problems.jsonl"
    rows = [
        {"problem_id": "p:一", "problem": "first"},
        {"problem_id": "p:2", "problem": "second"},
    ]
    records = {}
    with dataset.open("wb") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False).encode() + b"\n"
            offset = handle.tell()
            handle.write(payload)
            records[row["problem_id"]] = [offset, len(payload)]
    index = tmp_path / "problems.index.json"
    index.write_text(json.dumps({"records": records}), encoding="utf-8")

    store = IndexedProblemStore(dataset, index)
    assert store.get("p:2") == rows[1]
    assert store.get("p:一") == rows[0]
    with pytest.raises(KeyError, match="Problem not found"):
        store.get("missing")
