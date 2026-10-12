"""
hooks.py - Automatic capture for agents: Claude Code hook handlers and a generic command wrapper.
Recording no longer depends on the agent remembering to call MCP tools: every shell command is
checked before it runs and its outcome is logged after.
"""

import re
import subprocess
import sys
from typing import Any

from smruti.config import get_config
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.models import ActionStatus, InhibitionResult
from smruti.storage.db import get_db

_EXIT_CODE_RE = re.compile(r"^Exit code (\d+)")
_MAX_RESULT_CHARS = 2000


def _tail(text: str) -> str:
    return (text or "")[-_MAX_RESULT_CHARS:]


def preflight(command: str, project_root: str | None) -> InhibitionResult:
    db = get_db(get_config())
    return InhibitoryGate(db).check_action(command, project_root=project_root)


def record(command: str, result: str, failed: bool, session_id: str, project_root: str | None) -> None:
    config = get_config()
    stream = StreamBuffer(get_db(config), config=config, use_queue=False)
    stream.append(
        action=command,
        result=_tail(result),
        status=ActionStatus.FAILURE if failed else ActionStatus.SUCCESS,
        session_id=session_id,
        metadata={"project_root": project_root} if project_root else None,
    )
    # Short-lived process: let any triggered consolidation finish before exit
    stream.wait_for_consolidation(timeout=10.0)


def format_warning(res: InhibitionResult) -> str:
    fix = f" Suggested fix: {res.suggested_fix}" if res.suggested_fix else ""
    return f"smruti: this command matches a known past failure ({res.matched_signature}). {res.reason}{fix}"


def handle_claude_hook(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Handles a Claude Code hook event for the Bash tool. Returns the JSON to print, or None.
    PreToolUse: deny on blocking anti-memories, add context on advisory ones.
    PostToolUse / PostToolUseFailure: record the episode.
    """
    if payload.get("tool_name") != "Bash":
        return None
    command = (payload.get("tool_input") or {}).get("command", "").strip()
    if not command:
        return None
    event = payload.get("hook_event_name")
    cwd = payload.get("cwd")

    if event == "PreToolUse":
        res = preflight(command, cwd)
        if not res.passed:
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Blocked by smruti ({res.matched_signature}): {res.reason}",
            }}
        if res.matched_signature:
            # No permissionDecision: leave the user's normal permission flow untouched
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": format_warning(res)}}
        return None

    session_id = payload.get("session_id") or "claude_code"
    if event == "PostToolUse":
        response = payload.get("tool_response") or {}
        output = f"{response.get('stdout', '')}\n{response.get('stderr', '')}" if isinstance(response, dict) else str(response)
        record(command, output, failed=False, session_id=session_id, project_root=cwd)
    elif event == "PostToolUseFailure" and not payload.get("is_interrupt"):
        error = payload.get("error") or ""
        match = _EXIT_CODE_RE.match(error)
        failed = match is None or match.group(1) != "0"
        record(command, error, failed=failed, session_id=session_id, project_root=cwd)
    return None


def run_command(command: str, session_id: str, project_root: str | None) -> tuple[int, InhibitionResult]:
    """Generic wrapper: preflight, execute through the shell (output streams through), record outcome."""
    res = preflight(command, project_root)
    if not res.passed:
        return 1, res
    proc = subprocess.run(command, shell=True, cwd=project_root, stderr=subprocess.PIPE, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    record(command, proc.stderr or f"exit code {proc.returncode}", failed=proc.returncode != 0,
           session_id=session_id, project_root=project_root)
    return proc.returncode, res
