from __future__ import annotations

import re
from dataclasses import dataclass

from src.agent.schemas import ActionParseStatus, ActionType
from src.verifier import extract_code


_ACTION_PATTERN = re.compile(r"<action>\s*([^<]+?)\s*</action>", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedSubmission:
    requested_action: str | None
    action: ActionType
    parse_status: ActionParseStatus
    code: str | None


def parse_submission(response: str, *, execute_calls: int, max_execute_calls: int) -> ParsedSubmission:
    match = _ACTION_PATTERN.search(response)
    requested = match.group(1).strip().lower() if match else None
    if requested in {action.value for action in ActionType}:
        action = ActionType(requested)
        status = ActionParseStatus.EXPLICIT
    else:
        action = ActionType.EXECUTE_CODE if execute_calls < max_execute_calls else ActionType.FINAL
        status = (
            ActionParseStatus.MISSING_ACTION_FALLBACK
            if requested is None
            else ActionParseStatus.INVALID_ACTION_FALLBACK
        )
    return ParsedSubmission(
        requested_action=requested,
        action=action,
        parse_status=status,
        code=extract_code(response),
    )
