from scripts.postprocess_repair_api import (
    canonicalize_success,
    escalation_payload,
    recover_intermediate_success,
)


def _row(*, success: bool = True) -> dict:
    return {
        "task_id": "parent",
        "teacher_model": "qwen3-32b",
        "accepted": False,
        "rejection_reason": "invalid_action_protocol" if success else "repair_failed_full_tests",
        "repair_trajectory": {
            "termination_reason": "repeated_code",
            "outcome": {"final_success": success, "termination_reason": "repeated_code"},
            "steps": [
                {
                    "turn": 0,
                    "submission": {
                        "response": "```cpp\nint main(){}\n```",
                        "code": "int main(){}",
                        "effective_action": "execute_code",
                        "requested_action": None,
                        "action_parse_status": "missing_action_fallback",
                        "finish_reason": "stop",
                        "generation_tokens": 20,
                    },
                    "current_visible_pass_rate": 1.0,
                    "hidden_evaluation": {"judge": {"passed": int(success), "total": 1}},
                }
            ],
        },
    }


def test_canonicalize_verified_success_marks_final_without_changing_execution() -> None:
    value = canonicalize_success(_row(), source_run="run")
    submission = value["repair_trajectory"]["steps"][0]["submission"]
    assert submission["response"].startswith("<action>final</action>")
    assert submission["requested_action"] == submission["effective_action"] == "final"
    assert submission["action_parse_status"] == "explicit"
    assert value["normalization"]["execution_results_changed"] is False
    assert value["accepted"] is True


def test_escalation_uses_best_8b_candidate_as_new_initial_submission() -> None:
    row = _row(success=False)
    payload = escalation_payload(
        row,
        {"problem": {"problem_id": "p", "problem": "solve", "visible_tests": [1], "hidden_tests": [2]}},
        source_run="run",
    )
    assert payload is not None
    assert payload["problem"]["problem_id"] == "p"
    assert payload["initial_submission"]["producer_model"] == "qwen3-32b"
    assert payload["initial_submission"]["parent_task_id"] == "parent"


def test_recovers_full_pass_execute_before_later_regression() -> None:
    row = _row(success=False)
    first = row["repair_trajectory"]["steps"][0]
    first["prompt_messages"] = [{"role": "user", "content": "repair"}]
    first["submission"].update({"code_sha256": "abc", "prompt_tokens": None, "reasoning_content": None, "provider_metadata": {}})
    first["observation"] = {
        "judge": {"compiled": True, "passed": 1, "total": 1, "pass_rate": 1.0, "runtime_error": False, "timeout": False, "error_type": None, "compile_stderr": "", "cases": []}
    }
    first["hidden_evaluation"] = {
        "judge": {"compiled": True, "passed": 2, "total": 2, "pass_rate": 1.0, "runtime_error": False, "timeout": False, "error_type": None, "compile_stderr": "", "cases": []}
    }
    row["repair_trajectory"]["steps"].append(
        {
            "turn": 1,
            "prompt_messages": [{"role": "tool", "content": "passed"}],
            "submission": {**first["submission"], "turn": 1, "code": "bad", "code_sha256": "bad"},
            "observation": None,
            "hidden_evaluation": {"judge": {"passed": 0, "total": 3}},
            "previous_visible_pass_rate": 1.0,
            "current_visible_pass_rate": None,
            "delta_visible_pass_rate": None,
        }
    )
    value = recover_intermediate_success(row, source_run="run")
    assert value is not None
    assert len(value["repair_trajectory"]["steps"]) == 2
    final = value["repair_trajectory"]["steps"][-1]
    assert final["submission"]["effective_action"] == "final"
    assert final["submission"]["code"] == "int main(){}"
    assert value["repair_trajectory"]["hidden_evaluation"]["judge"]["total"] == 3
    assert value["normalization"]["truncated_turns"] == [1]
