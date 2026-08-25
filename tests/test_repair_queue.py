from __future__ import annotations

from src.data.repair_queue import RepairQueue


def test_repair_queue_is_idempotent_and_resumable(tmp_path) -> None:
    queue = RepairQueue(tmp_path / "tasks.sqlite3")
    assert queue.add("b", {"value": 2})
    assert queue.add("a", {"value": 1})
    assert not queue.add("a", {"value": 9})
    task_id, payload, attempts = queue.claim("worker")
    assert (task_id, payload, attempts) == ("a", {"value": 1}, 1)
    queue.complete(task_id, {"ok": True}, accepted=True)
    task_id, _, attempts = queue.claim("worker")
    assert task_id == "b" and attempts == 1
    queue.fail(task_id, "temporary", retry=True)
    task_id, _, attempts = queue.claim("worker-2")
    assert task_id == "b" and attempts == 2
    queue.complete(task_id, {"ok": False}, accepted=False)
    assert queue.counts() == {"accepted": 1, "rejected": 1}
    assert queue.export("accepted") == [{"ok": True}]
