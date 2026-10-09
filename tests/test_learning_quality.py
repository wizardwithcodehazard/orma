"""
test_learning_quality.py - Guards against the sleep cycle learning harmful memories:
over-broad blocks, stale failures, and spurious "fixes".
"""

import tempfile
from pathlib import Path

import pytest

from smruti.config import smrutiConfig
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.models import ActionStatus
from smruti.storage.db import DatabaseManager

FAIL = ActionStatus.FAILURE
OK = ActionStatus.SUCCESS


@pytest.fixture
def mem():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(project_dir=Path(tmpdir), db_filename="test_learning.db")
        config.auto_consolidate = False
        db = DatabaseManager(config)
        stream = StreamBuffer(db, config=config, auto_consolidate=False)
        gate = InhibitoryGate(db)
        cortex = Cortex(db, config)
        consolidator = Consolidator(db=db, config=config, stream=stream, inhibitory=gate, cortex=cortex)
        try:
            yield stream, gate, cortex, consolidator
        finally:
            db.close()


def test_learned_failures_warn_instead_of_block(mem):
    stream, gate, _, consolidator = mem
    for _ in range(2):
        stream.append("npm run build", result="ERR: ELIFECYCLE", status=FAIL)
    consolidator.sleep()

    res = gate.check_action("npm run build")
    assert res.passed is True
    assert res.matched_signature is not None
    assert res.severity == "warning"


def test_user_recorded_anti_memory_still_blocks(mem):
    _, gate, _, _ = mem
    gate.record_anti_memory(signature="no_rm_root", pattern="rm -rf /", reason="Catastrophic")
    assert gate.check_action("rm -rf /").passed is False


def test_unrelated_failures_do_not_cluster_by_first_word(mem):
    stream, gate, _, consolidator = mem
    stream.append("python train.py", result="ModuleNotFoundError: No module named 'torch'", status=FAIL)
    stream.append("python scripts/lint.py", result="SyntaxError: invalid syntax", status=FAIL)
    report = consolidator.sleep()

    assert report["promoted_anti_memories"] == 0
    for action in ["python --version", "python3 -m venv .venv"]:
        res = gate.check_action(action)
        assert res.passed is True and res.matched_signature is None


def test_failure_key_is_command_plus_error(mem):
    stream, gate, _, consolidator = mem
    for _ in range(2):
        stream.append("python  train.py", result="Traceback...\nModuleNotFoundError: No module named 'torch'", status=FAIL)
    consolidator.sleep()

    assert gate.check_action("python train.py --epochs 3").matched_signature is not None
    assert gate.check_action("python --version").matched_signature is None
    # Context text mentioning the command must not trigger a match
    assert gate.check_action("ls", context="earlier python train.py failed").matched_signature is None
