from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class IndexedProblemStore:
    """Thread-safe random access to a JSONL problem dataset via byte offsets."""

    def __init__(self, dataset: str | Path, index: str | Path) -> None:
        self.dataset = Path(dataset)
        payload = json.loads(Path(index).read_text(encoding="utf-8"))
        self.records: dict[str, list[int]] = payload["records"]
        self._local = threading.local()

    def _handle(self):
        handle = getattr(self._local, "handle", None)
        if handle is None or handle.closed:
            handle = self.dataset.open("rb")
            self._local.handle = handle
        return handle

    def get(self, problem_id: str) -> dict[str, Any]:
        try:
            offset, length = self.records[problem_id]
        except KeyError as error:
            raise KeyError(f"Problem not found in compact dataset: {problem_id}") from error
        handle = self._handle()
        handle.seek(offset)
        payload = handle.read(length)
        row = json.loads(payload)
        if row.get("problem_id") != problem_id:
            raise ValueError(f"Problem index mismatch at offset {offset}: {problem_id}")
        return row
