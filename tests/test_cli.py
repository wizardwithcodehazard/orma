import json

import pytest
from typer.testing import CliRunner

from smruti import Memory, cli

runner = CliRunner()


@pytest.fixture
def seeded(config, embedder, monkeypatch, seed_state):
    monkeypatch.setattr(cli, "_memory", lambda upstream=None, model=None: Memory(config=config, embedder=embedder))
    mem = Memory(config=config, embedder=embedder)
    state_id = seed_state(mem.store, "default", "home_city", "Pune", text="I live in Pune")
    mem.close()
    return state_id


def test_help_lists_commands():
    out = runner.invoke(cli.app, ["--help"]).output
    for command in ("add", "search", "proxy", "sleep", "rebuild", "facts", "recall", "forget", "stats"):
        assert command in out


def test_facts_show_the_sentence_and_forget_ends_the_state(seeded):
    out = runner.invoke(cli.app, ["facts"]).output
    assert "user · home city: Pune" in out and "user: I live in Pune" in out
    result = runner.invoke(cli.app, ["forget", str(seeded)])
    assert result.exit_code == 0 and "Forgotten" in result.output
    assert "home city" not in runner.invoke(cli.app, ["facts"]).output
    assert "(until " in runner.invoke(cli.app, ["facts", "--history"]).output
    assert runner.invoke(cli.app, ["forget", str(seeded)]).exit_code == 1


def test_add_text_and_file_then_search(seeded, tmp_path):
    assert "Kept 1 of 1 records" in runner.invoke(cli.app, ["add", "My dentist is Dr. Rao in Kothrud."]).output
    notes = tmp_path / "trip.md"
    notes.write_text("Packing list for Manali: jacket, boots.\n\nTrain leaves at 6 am.", encoding="utf-8")
    assert "Kept 1 of 1 records" in runner.invoke(cli.app, ["add", str(notes)]).output
    out = runner.invoke(cli.app, ["search", "dentist"]).output
    assert "Things said:" in out and "My dentist is Dr. Rao in Kothrud." in out
    assert "trip.md: Train leaves at 6 am." in runner.invoke(cli.app, ["search", "train time"]).output


def test_recall_prints_the_memory_block(seeded):
    out = runner.invoke(cli.app, ["recall", "where do I live?"]).output
    assert "## Memory" in out and "user · home city: Pune" in out


def test_stats_prints_json(seeded):
    data = json.loads(runner.invoke(cli.app, ["stats"]).output)
    assert data["space"] == "default" and data["current_states"] == 1 and data["sentences"] == 1


def test_proxy_quiets_http_client_loggers(seeded, monkeypatch):
    import logging
    import sys
    import types

    import smruti.serve.proxy as proxy_module

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    monkeypatch.setattr(proxy_module, "create_app", lambda *a, **k: object())
    saved = {name: logging.getLogger(name).level for name in ("httpx", "httpcore", "smruti")}
    try:
        runner.invoke(cli.app, ["proxy"])
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("smruti").level == saved["smruti"]
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
