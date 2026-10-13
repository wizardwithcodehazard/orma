import json

import pytest
from typer.testing import CliRunner

import smruti.cli as cli
from smruti import Memory

runner = CliRunner()


@pytest.fixture
def seeded(config, embedder, monkeypatch):
    monkeypatch.setattr(cli, "_memory", lambda upstream=None, model=None: Memory(config=config, embedder=embedder))
    mem = Memory(config=config, embedder=embedder)
    s = mem.store
    fact_id = s.add_fact("default", s.user_entity("default"), s.add_predicate("default", "lives_in", "one"),
                         s.add_entity("default", "Pune", "city"), None, 1.0, [1])
    mem.close()
    return fact_id


def test_help_lists_commands():
    out = runner.invoke(cli.app, ["--help"]).output
    for command in ("proxy", "sleep", "facts", "recall", "forget", "stats"):
        assert command in out


def test_facts_forget_and_history(seeded):
    assert "user lives in Pune" in runner.invoke(cli.app, ["facts"]).output
    result = runner.invoke(cli.app, ["forget", str(seeded)])
    assert result.exit_code == 0 and "Forgotten" in result.output
    assert "user lives in Pune" not in runner.invoke(cli.app, ["facts"]).output
    assert "no longer true" in runner.invoke(cli.app, ["facts", "--history"]).output
    assert runner.invoke(cli.app, ["forget", str(seeded)]).exit_code == 1


def test_recall_prints_the_memory_block(seeded):
    out = runner.invoke(cli.app, ["recall", "where do I live?"]).output
    assert "## Memory" in out and "user lives in Pune" in out


def test_stats_prints_json(seeded):
    data = json.loads(runner.invoke(cli.app, ["stats"]).output)
    assert data["space"] == "default" and data["current_facts"] == 1
