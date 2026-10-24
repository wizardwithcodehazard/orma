from pathlib import Path

from orma.config import SmrutiConfig


def test_db_path_lives_in_data_dir(tmp_path):
    assert SmrutiConfig(data_dir=tmp_path).db_path == tmp_path / "memory.db"


def test_data_dir_comes_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SMRUTI_DIR", str(tmp_path))
    assert SmrutiConfig().data_dir == tmp_path.resolve()


def test_data_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("ORMA_DIR", raising=False)
    monkeypatch.delenv("SMRUTI_DIR", raising=False)
    assert SmrutiConfig().data_dir == Path.home() / ".orma"


def test_citable_and_recalled_roles_leave_out_the_assistant():
    config = SmrutiConfig()
    assert "assistant" not in config.citable_roles and "assistant" not in config.recall_roles
