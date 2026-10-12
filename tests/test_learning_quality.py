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


def test_learned_failure_retired_when_command_later_succeeds(mem):
    stream, gate, _, consolidator = mem
    for _ in range(2):
        stream.append("python train.py", result="ModuleNotFoundError: No module named 'torch'", status=FAIL)
    consolidator.sleep()
    assert gate.check_action("python train.py").matched_signature is not None

    stream.append("pip install torch", result="Successfully installed torch", status=OK)
    stream.append("python train.py", result="epoch 1 done", status=OK)
    report = consolidator.sleep()

    assert report["retired_anti_memories"] == 1
    assert gate.check_action("python train.py").matched_signature is None


def test_cluster_resolved_in_same_batch_is_not_promoted(mem):
    stream, gate, _, consolidator = mem
    for _ in range(2):
        stream.append("npm run build", result="ERR: ELIFECYCLE", status=FAIL)
    stream.append("npm run build", result="built in 2s", status=OK)
    report = consolidator.sleep()

    assert report["promoted_anti_memories"] == 0
    assert gate.check_action("npm run build").matched_signature is None


def test_learned_failures_expire(mem):
    stream, gate, _, consolidator = mem
    for _ in range(2):
        stream.append("make deploy", result="permission denied", status=FAIL)
    consolidator.sleep()
    (anti,) = [a for a in gate.list_all() if a.pattern == "make deploy"]
    assert anti.expires_at is not None


def test_unrelated_success_is_not_learned_as_fix(mem):
    stream, _, cortex, consolidator = mem
    stream.append("python train.py", result="ModuleNotFoundError: No module named 'torch'", status=FAIL)
    stream.append("cat README.md", result="# Project", status=OK)          # read-only, not a fix
    stream.append("git commit -m wip", result="1 file changed", status=OK)  # unrelated
    report = consolidator.sleep()

    assert report["promoted_positive_rules"] == 0
    assert cortex.count() == 0


def test_related_fix_is_learned_tentatively_then_strengthened(mem):
    stream, _, cortex, consolidator = mem
    stream.append("python train.py", result="ModuleNotFoundError: No module named 'torch'", status=FAIL)
    stream.append("cat README.md", result="# Project", status=OK)
    stream.append("pip install torch", result="Successfully installed torch", status=OK)
    consolidator.sleep()

    (rule,) = [r for r, _ in cortex.recall_rules("torch")]
    assert "pip install torch" in rule.rule_text
    assert rule.confidence == 0.5

    stream.append("python train.py", result="ModuleNotFoundError: No module named 'torch'", status=FAIL)
    stream.append("pip install torch", result="Successfully installed torch", status=OK)
    consolidator.sleep()

    (rule,) = [r for r, _ in cortex.recall_rules("torch")]
    assert rule.confidence > 0.5


def test_distinct_learned_rules_are_not_merged(mem):
    stream, _, cortex, consolidator = mem
    stream.append("npm run build", result="Cannot find module 'vite'", status=FAIL)
    stream.append("npm install vite", result="added 1 package", status=OK)
    stream.append("pytest tests/", result="No module named 'requests'", status=FAIL)
    stream.append("pip install requests", result="Successfully installed requests", status=OK)
    consolidator.sleep()

    assert cortex.count() == 2


def test_user_rules_never_decay_but_learned_rules_do(mem):
    _, _, cortex, _ = mem
    from smruti.models import LEARNED_RULE_CATEGORY

    cortex.add_rule("Never deploy on Friday afternoon", category="devops")
    learned = cortex.add_rule("When 'make' fails, 'make clean' resolved it.", category=LEARNED_RULE_CATEGORY, base_strength=0.5)
    month_later = learned.created_at + 30 * 86400

    assert cortex.prune_decayed_rules(current_time=month_later) == 1
    (survivor,) = [r for r, _ in cortex.recall_rules("deploy")]
    assert survivor.rule_text == "Never deploy on Friday afternoon"


def test_recall_does_not_strengthen_rules(mem):
    _, _, cortex, _ = mem
    rule = cortex.add_rule("Use uv for dependency management", category="tooling", base_strength=0.6)
    for _ in range(5):
        cortex.recall_rules("dependency management")
    (after,) = [r for r, _ in cortex.recall_rules("dependency management")]
    assert after.base_strength == rule.base_strength


def test_program_name_alone_is_not_evidence_of_a_fix(mem):
    stream, _, cortex, consolidator = mem
    stream.append("python train.py", result="ModuleNotFoundError: No module named 'torch'", status=FAIL)
    stream.append("python --version", result="Python 3.12.4", status=OK)
    consolidator.sleep()
    assert cortex.count() == 0


def test_failures_and_fixes_pair_across_sleep_cycles(mem):
    stream, gate, cortex, consolidator = mem
    stream.append("npm run build", result="Cannot find module 'vite'", status=FAIL)
    consolidator.sleep()
    stream.append("npm run build", result="Cannot find module 'vite'", status=FAIL)
    consolidator.sleep()
    assert gate.check_action("npm run build").matched_signature is not None  # counted across cycles

    stream.append("npm install vite", result="added 1 package", status=OK)
    consolidator.sleep()
    assert cortex.count() == 1  # fix paired with a failure from an earlier cycle

    consolidator.sleep()
    (rule,) = [r for r, _ in cortex.recall_rules("vite")]
    assert rule.confidence == 0.5  # re-running sleep does not re-count old pairs
