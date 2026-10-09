"""
test_concurrency_and_retries.py - Unit tests for:
1. PRAGMA busy_timeout = 5000 on connection creation
2. Exponential backoff and jitter retry handler for write transactions
3. In-memory queue worker thread for episodic buffer writes under parallel multi-agent execution
"""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from smruti.config import smrutiConfig
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.models import ActionStatus
from smruti.storage.db import DatabaseManager, calculate_backoff, is_db_locked_error


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_concurrency.db",
            busy_timeout_ms=5000,
            write_max_retries=5,
            write_base_delay=0.02,
            write_max_delay=0.5
        )
        db = DatabaseManager(config)
        try:
            yield db, config
        finally:
            db.close()


def test_pragma_busy_timeout_on_connection(temp_db):
    """Verifies that PRAGMA busy_timeout is configured to 5000 on connection creation."""
    db, _ = temp_db
    conn = db.get_connection()
    row = conn.execute("PRAGMA busy_timeout;").fetchone()
    assert row[0] == 5000, f"Expected busy_timeout 5000, got {row[0]}"


def test_backoff_and_jitter_calculation():
    """Verifies exponential backoff scaling and random jitter bounds."""
    for attempt in range(5):
        delay = calculate_backoff(attempt, base_delay=0.02, max_delay=0.5)
        nominal = min(0.5, 0.02 * (2 ** attempt))
        assert 0.5 * nominal <= delay <= nominal, f"Jitter out of bounds for attempt {attempt}: {delay}"


def test_is_db_locked_error_detection():
    """Verifies that OperationalError with 'database is locked' or 'busy' is identified."""
    e_locked = sqlite3.OperationalError("database is locked")
    e_busy = sqlite3.OperationalError("database is busy")
    e_syntax = sqlite3.OperationalError("near 'SYNTAX': syntax error")
    assert is_db_locked_error(e_locked) is True
    assert is_db_locked_error(e_busy) is True
    assert is_db_locked_error(e_syntax) is False
    assert is_db_locked_error(ValueError("other error")) is False


def test_retry_handler_recovers_from_transient_lock(temp_db):
    """
    Verifies that execute_write and write_transaction retry with backoff and succeed
    when another connection holds a transient lock.
    """
    db, config = temp_db

    # Open a separate independent connection that holds an exclusive lock for 60ms
    def locker():
        ext_conn = sqlite3.connect(str(config.db_path), timeout=0.1)
        ext_conn.execute("BEGIN EXCLUSIVE;")
        time.sleep(0.06)
        ext_conn.commit()
        ext_conn.close()

    lock_thread = threading.Thread(target=locker)
    lock_thread.start()
    # Ensure locker acquired exclusive lock
    time.sleep(0.01)

    # execute_write should back off, retry, and succeed once the lock is released
    start = time.perf_counter()
    res = db.execute_write(
        "INSERT INTO episodes (id, timestamp, session_id, action, status, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
        ("retry_test_1", time.time(), "s1", "cmd", "success", 1.0),
        max_retries=5,
        base_delay=0.02,
        max_delay=0.2
    )
    elapsed = time.perf_counter() - start
    lock_thread.join()

    assert res.rowcount == 1
    assert elapsed >= 0.04, f"Expected retry delay, took {elapsed:.4f}s"


def test_retry_handler_exhausts_and_raises(temp_db):
    """
    Verifies that if the lock is held longer than the max retries allow,
    the retry handler re-raises sqlite3.OperationalError.
    """
    db, config = temp_db

    ext_conn = sqlite3.connect(str(config.db_path), timeout=0.1)
    ext_conn.execute("PRAGMA busy_timeout = 0;")
    ext_conn.execute("BEGIN EXCLUSIVE;")

    try:
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            db.execute_write(
                "INSERT INTO episodes (id, timestamp, session_id, action, status, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                ("exhaust_test", time.time(), "s1", "cmd", "success", 1.0),
                max_retries=3,
                base_delay=0.01,
                max_delay=0.03
            )
        assert "locked" in str(exc_info.value).lower() or "busy" in str(exc_info.value).lower()
    finally:
        ext_conn.rollback()
        ext_conn.close()


def test_multi_agent_parallel_stream_appends(temp_db):
    """
    Simulates multi-agent execution with 20 parallel threads calling stream.append concurrently.
    Verifies zero lock errors and complete commit integrity.
    """
    db, _ = temp_db
    stream = StreamBuffer(db, auto_consolidate=False, use_queue=True)

    thread_count = 20
    appends_per_thread = 25
    total_episodes = thread_count * appends_per_thread

    errors = []

    def agent_worker(agent_id: int):
        for i in range(appends_per_thread):
            try:
                stream.append(
                    action=f"agent_{agent_id}_step_{i}",
                    result="ok",
                    status=ActionStatus.SUCCESS,
                    session_id=f"agent_{agent_id}"
                )
            except Exception as e:
                errors.append((agent_id, i, e))

    threads = [threading.Thread(target=agent_worker, args=(t,)) for t in range(thread_count)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    assert len(errors) == 0, f"Encountered {len(errors)} errors during parallel appends: {errors[:3]}"
    assert stream.count() == total_episodes
    print(f"\nMulti-agent parallel benchmark: {total_episodes} appends across {thread_count} threads in {elapsed:.3f}s")


def test_stream_async_and_flush(temp_db):
    """Verifies non-blocking async writes (sync=False) and explicit queue flush."""
    db, _ = temp_db
    stream = StreamBuffer(db, auto_consolidate=False, use_queue=True)

    for i in range(50):
        stream.append(
            action=f"async_action_{i}",
            result="done",
            status=ActionStatus.SUCCESS,
            session_id="async_test",
            sync=False
        )

    # Flush drains queue
    stream.flush()
    assert stream.count(session_id="async_test") == 50


def test_concurrent_stream_writes_with_active_consolidation(temp_db):
    """
    Verifies that parallel episodic stream writes and active consolidation sleep cycles
    coexist safely without raising 'database is locked'.
    """
    db, config = temp_db
    stream = StreamBuffer(db, auto_consolidate=False, use_queue=True)
    gate = InhibitoryGate(db)
    cortex = Cortex(db, config)
    consolidator = Consolidator(db, config, stream=stream, inhibitory=gate, cortex=cortex)

    stop_flag = threading.Event()
    consolidation_errors = []
    append_errors = []

    def consolidator_loop():
        while not stop_flag.is_set():
            try:
                consolidator.sleep()
                time.sleep(0.01)
            except Exception as e:
                consolidation_errors.append(e)

    cons_thread = threading.Thread(target=consolidator_loop)
    cons_thread.start()

    def appender(agent_id: int):
        for i in range(30):
            try:
                stream.append(
                    action=f"cmd_{agent_id}_{i}",
                    result="ok",
                    status=ActionStatus.SUCCESS,
                    session_id=f"agent_{agent_id}"
                )
            except Exception as e:
                append_errors.append(e)

    appender_threads = [threading.Thread(target=appender, args=(a,)) for a in range(8)]
    for t in appender_threads:
        t.start()
    for t in appender_threads:
        t.join()

    stop_flag.set()
    cons_thread.join()

    assert len(append_errors) == 0, f"Append errors: {append_errors}"
    assert len(consolidation_errors) == 0, f"Consolidation errors: {consolidation_errors}"
