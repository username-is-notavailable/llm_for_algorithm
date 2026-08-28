from __future__ import annotations

import json

import pytest

from scripts.prepare_codecontests_plus_repair_resumable import (
    atomic_write_json,
    config_digest,
    open_outputs,
    sync_checkpoint,
    write_record,
)


def config(tmp_path):
    return {
        "output": {
            "dataset": str(tmp_path / "problems.jsonl"),
            "failure_pool": str(tmp_path / "failure_pool.jsonl"),
            "one_shot_seeds": str(tmp_path / "one_shot_seeds.jsonl"),
            "checkpoint": str(tmp_path / "prepare.checkpoint.json"),
        }
    }


def test_resume_truncates_uncommitted_appends(tmp_path) -> None:
    value = config(tmp_path)
    state, handles, files, state_path = open_outputs(value, resume=False)
    for handle in handles.values():
        write_record(handle, {"committed": True})
    state["accepted"] = 1
    state["scanned_candidates"] = 7
    sync_checkpoint(state_path, state, handles)
    for handle in handles.values():
        write_record(handle, {"must_be_truncated": True})
        handle.flush()
        handle.close()

    resumed, resumed_handles, _, _ = open_outputs(value, resume=True)
    try:
        assert resumed["accepted"] == 1
        for name, handle in resumed_handles.items():
            handle.flush()
            assert files[name].read_text(encoding="utf-8") == '{"committed":true}\n'
    finally:
        for handle in resumed_handles.values():
            handle.close()


def test_resume_rejects_changed_config(tmp_path) -> None:
    value = config(tmp_path)
    _, handles, _, state_path = open_outputs(value, resume=False)
    for handle in handles.values():
        handle.close()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    value["new_setting"] = True
    assert state["config_sha256"] != config_digest(value)
    with pytest.raises(ValueError, match="Checkpoint config differs"):
        open_outputs(value, resume=True)


def test_fresh_run_refuses_to_overwrite_partial_data(tmp_path) -> None:
    value = config(tmp_path)
    _, handles, _, _ = open_outputs(value, resume=False)
    for handle in handles.values():
        handle.close()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        open_outputs(value, resume=False)
