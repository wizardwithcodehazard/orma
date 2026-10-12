"""
test_hooks.py - Claude Code hook handlers and the `smruti run` wrapper record and gate automatically.
"""

import pytest
from typer.testing import CliRunner

from smruti.config import get_config
from smruti.engine.consolidator import Consolidator
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.interfaces.cli import app
from smruti.interfaces.hooks import handle_claude_hook
from smruti.storage.db import close_all_dbs, get_db


@pytest.fixture(autouse=True)
def isolated_smruti(tmp_path, monkeypatch):
    monkeypatch.setenv("SMRUTI_DIR", str(tmp_path / ".smruti"))
    yield tmp_path
    close_all_dbs()


def _bash(event, command, cwd="/proj", **extra):
    return {"hook_event_name": event, "tool_name": "Bash", "tool_input": {"command": command},
            "cwd": cwd, "session_id": "s1", **extra}


def test_post_hooks_record_success_and_failure():
    handle_claude_hook(_bash("PostToolUse", "ls", tool_response={"stdout": "a.py", "stderr": ""}))
    handle_claude_hook(_bash("PostToolUseFailure", "npm run build", error="Exit code 1\nERR: ELIFECYCLE"))
    handle_claude_hook(_bash("PostToolUseFailure", "sleep 100", error="Interrupted", is_interrupt=True))

    eps = {e.action: e for e in StreamBuffer(get_db(get_config()), use_queue=False).get_recent()}
    assert set(eps) == {"ls", "npm run build"}
    assert eps["npm run build"].status.value == "failure"
    assert eps["npm run build"].metadata["project_root"] == "/proj"


def test_pre_hook_warns_on_learned_failure_without_touching_permissions():
    for _ in range(2):
        handle_claude_hook(_bash("PostToolUseFailure", "npm run build", error="Exit code 1\nERR: ELIFECYCLE"))
    Consolidator(db=get_db(get_config())).sleep()

    out = handle_claude_hook(_bash("PreToolUse", "npm run build"))
    spec = out["hookSpecificOutput"]
    assert "permissionDecision" not in spec
    assert "known past failure" in spec["additionalContext"]
    assert handle_claude_hook(_bash("PreToolUse", "npm test")) is None


def test_pre_hook_denies_user_recorded_anti_memory():
    InhibitoryGate(get_db(get_config())).record_anti_memory("no_force_push", "git push --force", "Protected branch")
    out = handle_claude_hook(_bash("PreToolUse", "git push --force origin main"))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_non_bash_tools_are_ignored():
    assert handle_claude_hook({"hook_event_name": "PreToolUse", "tool_name": "Edit", "tool_input": {}}) is None


def test_cli_hook_fails_open_on_bad_input():
    res = CliRunner().invoke(app, ["hook"], input="not json")
    assert res.exit_code == 0 and res.stdout == ""


def test_cli_run_records_exit_status(isolated_smruti):
    runner = CliRunner()
    assert runner.invoke(app, ["run", "--", "true"]).exit_code == 0
    assert runner.invoke(app, ["run", "--", "false"]).exit_code == 1

    eps = {e.action: e.status.value for e in StreamBuffer(get_db(get_config()), use_queue=False).get_recent()}
    assert eps == {"true": "success", "false": "failure"}
