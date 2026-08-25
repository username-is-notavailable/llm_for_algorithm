from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepairQueue:
    """Small persistent work queue with atomic leases and resumable results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def _initialize(self) -> None:
        self._connection().execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT,
                leased_at TEXT,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def add(self, task_id: str, payload: dict[str, Any]) -> bool:
        now = utc_now()
        cursor = self._connection().execute(
            """INSERT OR IGNORE INTO tasks(task_id,payload,created_at,updated_at)
               VALUES(?,?,?,?)""",
            (task_id, json.dumps(payload, ensure_ascii=False), now, now),
        )
        return cursor.rowcount == 1

    def reclaim_stale(self, lease_timeout_seconds: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=lease_timeout_seconds)).isoformat()
        cursor = self._connection().execute(
            """UPDATE tasks SET status='pending',worker_id=NULL,leased_at=NULL,updated_at=?
               WHERE status='running' AND leased_at < ?""",
            (utc_now(), cutoff),
        )
        return cursor.rowcount

    def claim(self, worker_id: str) -> tuple[str, dict[str, Any], int] | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT task_id,payload FROM tasks WHERE status='pending' ORDER BY task_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = utc_now()
            connection.execute(
                """UPDATE tasks SET status='running',attempts=attempts+1,worker_id=?,
                   leased_at=?,updated_at=? WHERE task_id=?""",
                (worker_id, now, now, row["task_id"]),
            )
            attempts = int(
                connection.execute(
                    "SELECT attempts FROM tasks WHERE task_id=?", (row["task_id"],)
                ).fetchone()["attempts"]
            )
            return str(row["task_id"]), json.loads(row["payload"]), attempts

    def complete(self, task_id: str, result: dict[str, Any], *, accepted: bool) -> None:
        self._connection().execute(
            """UPDATE tasks SET status=?,result=?,error=NULL,updated_at=? WHERE task_id=?""",
            (
                "accepted" if accepted else "rejected",
                json.dumps(result, ensure_ascii=False),
                utc_now(),
                task_id,
            ),
        )

    def fail(self, task_id: str, error: str, *, retry: bool) -> None:
        self._connection().execute(
            """UPDATE tasks SET status=?,error=?,worker_id=NULL,leased_at=NULL,updated_at=?
               WHERE task_id=?""",
            ("pending" if retry else "failed", error, utc_now(), task_id),
        )

    def counts(self) -> dict[str, int]:
        return {
            str(row["status"]): int(row["count"])
            for row in self._connection().execute(
                "SELECT status,COUNT(*) AS count FROM tasks GROUP BY status"
            )
        }

    def export(self, status: str) -> list[dict[str, Any]]:
        rows = self._connection().execute(
            "SELECT result FROM tasks WHERE status=? ORDER BY task_id", (status,)
        )
        return [json.loads(row["result"]) for row in rows if row["result"]]
