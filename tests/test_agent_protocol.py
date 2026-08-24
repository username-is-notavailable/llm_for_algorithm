from src.agent import ActionParseStatus, ActionType, AgentConfig, parse_submission


CODE = """```cpp
int main() { return 0; }
```"""


def test_parses_explicit_execute_and_final_actions() -> None:
    execute = parse_submission(
        f"<action>execute_code</action>\n{CODE}", execute_calls=0, max_execute_calls=3
    )
    final = parse_submission(
        f"<action>final</action>\n{CODE}", execute_calls=1, max_execute_calls=3
    )
    assert execute.action == ActionType.EXECUTE_CODE
    assert final.action == ActionType.FINAL
    assert execute.parse_status == ActionParseStatus.EXPLICIT
    assert execute.code == "int main() { return 0; }"


def test_missing_or_invalid_action_uses_budget_aware_fallback() -> None:
    missing = parse_submission(CODE, execute_calls=1, max_execute_calls=3)
    exhausted = parse_submission(CODE, execute_calls=3, max_execute_calls=3)
    invalid = parse_submission(
        f"<action>shell</action>\n{CODE}", execute_calls=0, max_execute_calls=3
    )
    assert missing.action == ActionType.EXECUTE_CODE
    assert missing.parse_status == ActionParseStatus.MISSING_ACTION_FALLBACK
    assert exhausted.action == ActionType.FINAL
    assert invalid.action == ActionType.EXECUTE_CODE
    assert invalid.parse_status == ActionParseStatus.INVALID_ACTION_FALLBACK


def test_agent_config_reserves_a_final_candidate_slot() -> None:
    AgentConfig(max_execute_calls=3, max_candidate_submissions=4)
    try:
        AgentConfig(max_execute_calls=3, max_candidate_submissions=3)
    except ValueError as error:
        assert "final submission" in str(error)
    else:
        raise AssertionError("Expected invalid candidate budget to fail")
