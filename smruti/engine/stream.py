"""
stream.py - Tier 1: Sub-millisecond Episodic Working Buffer (Hippocampal stream).
Append-only, transaction-safe SQLite WAL stream with zero LLM overhead.
Supports unconsolidated queries, batch consolidation marking, retention sweeps,
and autonomic background sleep cycles triggered by turn thresholds.
"""

import atexit
import json
import logging
import queue
import re
import sqlite3
import threading
import time
from typing import Any

from smruti.config import smrutiConfig, get_config
from smruti.models import ActionStatus, Episode
from smruti.storage.db import DatabaseManager, get_db

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = [
    # Generic API Keys (OpenAI, Anthropic, Stripe, etc.)
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    # GitHub Tokens
    (re.compile(r"(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{50,}"), "[REDACTED_GITHUB_TOKEN]"),
    # Bearer Tokens
    (re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{20,}"), r"\1[REDACTED_BEARER_TOKEN]"),
    # AWS Access Key ID
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    # Passwords in URLs / connection strings
    (re.compile(r"(://[^:]+:)[^@]+(@)"), r"\1[REDACTED_PASSWORD]\2"),
    # Private Keys
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
]

def mask_sensitive_data(text: str) -> str:
    """Masks secrets, credentials, and authentication tokens before persisting to disk."""
    if not text:
        return text
    masked = text
    for pattern, replacement in _SECRET_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked

class StreamBuffer:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        config: smrutiConfig | None = None,
        auto_consolidate: bool = True,
        use_queue: bool | None = None
    ):
        self.config = config or get_config()
        self.db = db or get_db(self.config)
        self.auto_consolidate = auto_consolidate and self.config.auto_consolidate
        self.use_queue = use_queue if use_queue is not None else getattr(self.config, "use_queue_worker", True)
        self._consolidating = False
        self._consolidate_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._closed = False
        self._worker_thread: threading.Thread | None = None

        if self.use_queue:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="smruti-stream-writer",
                daemon=True
            )
            self._worker_thread.start()
            if hasattr(self.db, "register_stream"):
                self.db.register_stream(self)

    def _worker_loop(self) -> None:
        """Dedicated background writer thread that consumes queued episodes and writes in batches."""
        batch_size = getattr(self.config, "queue_batch_size", 100)
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.02)
            except queue.Empty:
                continue

            batch = [item]
            while len(batch) < batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                self._write_batch(batch)
            except Exception as e:
                logger.error("Error writing episode batch in queue worker: %s", e)
            finally:
                for _ in batch:
                    self._queue.task_done()

        # Drain any remaining items after stop_event
        remaining = []
        while True:
            try:
                remaining.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if remaining:
            try:
                self._write_batch(remaining)
            except Exception as e:
                logger.error("Error writing remaining episode batch in queue worker: %s", e)
            finally:
                for _ in remaining:
                    self._queue.task_done()

    def _write_batch(self, batch: list[tuple[Episode, threading.Event | None, list]]) -> None:
        if not batch:
            return

        def _insert_all(conn: sqlite3.Connection):
            conn.executemany(
                """
                INSERT INTO episodes (
                    id, timestamp, session_id, context, action, result, status,
                    latency_ms, metadata_json, consolidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ep.id,
                        ep.timestamp,
                        ep.session_id,
                        ep.context,
                        ep.action,
                        ep.result,
                        ep.status.value,
                        ep.latency_ms,
                        json.dumps(ep.metadata),
                        None
                    )
                    for ep, _, _ in batch
                ]
            )

        err = None
        try:
            self.db.execute_write(_insert_all)
        except Exception as e:
            err = e
            raise
        finally:
            for _, event, err_container in batch:
                if event is not None:
                    if err is not None:
                        err_container.append(err)
                    event.set()

        # Trigger autonomic background consolidation if threshold reached
        if self.auto_consolidate:
            sessions = {ep.session_id for ep, _, _ in batch}
            for sid in sessions:
                self._check_and_trigger_auto_consolidation(sid)

    def flush(self, timeout: float | None = 5.0) -> None:
        """Blocks until all queued episodes have been persisted to SQLite."""
        if not self.use_queue:
            return
        if self._queue.unfinished_tasks > 0:
            self._queue.join()

    def close(self, timeout: float = 1.0) -> None:
        """Flushes pending writes and stops the queue worker thread."""
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            if threading.current_thread() != self._worker_thread:
                self._worker_thread.join(timeout=timeout)

    def append(
        self,
        action: str,
        result: str = "",
        status: ActionStatus = ActionStatus.SUCCESS,
        context: str = "",
        latency_ms: float = 0.0,
        session_id: str = "default_session",
        metadata: dict[str, Any] | None = None,
        sync: bool = True
    ) -> Episode:
        """
        Appends an action trajectory into the episodic buffer in sub-millisecond time.
        Zero LLM latency, zero graph blocking.
        Automatically sanitizes secrets, tokens, and credentials.
        Autonomously fires a non-blocking background consolidation cycle if the
        unconsolidated turn threshold is reached.
        """
        clean_action = mask_sensitive_data(action)
        clean_result = mask_sensitive_data(result)
        clean_context = mask_sensitive_data(context)

        episode = Episode(
            session_id=session_id,
            context=clean_context,
            action=clean_action,
            result=clean_result,
            status=status,
            latency_ms=latency_ms,
            metadata=metadata or {},
            consolidated_at=None
        )

        if self.use_queue and not self._closed:
            if sync:
                event = threading.Event()
                err_container = []
                self._queue.put((episode, event, err_container))
                event.wait()
                if err_container:
                    raise err_container[0]
            else:
                self._queue.put((episode, None, []))
            return episode

        # Direct write path (when queue disabled or stream closed)
        def _insert(conn: sqlite3.Connection):
            conn.execute(
                """
                INSERT INTO episodes (
                    id, timestamp, session_id, context, action, result, status,
                    latency_ms, metadata_json, consolidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.timestamp,
                    episode.session_id,
                    episode.context,
                    episode.action,
                    episode.result,
                    episode.status.value,
                    episode.latency_ms,
                    json.dumps(episode.metadata),
                    None
                )
            )

        self.db.execute_write(_insert)

        # Autonomic background sleep threshold check
        if self.auto_consolidate:
            self._check_and_trigger_auto_consolidation(session_id)

        return episode

    def _check_and_trigger_auto_consolidation(self, session_id: str | None = None) -> None:
        """Checks if unconsolidated episode count exceeds threshold and fires background worker."""
        try:
            conn = self.db.get_connection()
            cur = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated_at IS NULL")
            unconsolidated_count = cur.fetchone()[0]
            if unconsolidated_count >= self.config.consolidation_turn_interval:
                self._start_background_consolidation(session_id)
        except Exception as e:
            logger.debug("Auto-consolidation threshold check error: %s", e)

    def _start_background_consolidation(self, session_id: str | None = None) -> None:
        """Launches a daemon worker thread to run consolidation without blocking write path."""
        with self._consolidate_lock:
            if self._consolidating:
                return
            self._consolidating = True

        def _worker():
            try:
                from smruti.engine.consolidator import Consolidator
                consolidator = Consolidator(db=self.db, config=self.config, stream=self)
                report = consolidator.sleep(session_id=session_id)
                logger.info(
                    "Autonomic background consolidation complete: %d episodes, %d anti-memories, %d rules",
                    report.get("processed_episodes", 0),
                    report.get("promoted_anti_memories", 0),
                    report.get("promoted_positive_rules", 0)
                )
            except Exception as e:
                logger.warning("Background consolidation failed: %s", e)
            finally:
                with self._consolidate_lock:
                    self._consolidating = False

        thread = threading.Thread(target=_worker, name="smrutiAutonomicSleep", daemon=True)
        self._active_worker = thread
        thread.start()

    def wait_for_consolidation(self, timeout: float = 3.0) -> None:
        """Blocks until the active background consolidation completes, up to timeout seconds."""
        if hasattr(self, "_active_worker") and self._active_worker is not None and self._active_worker.is_alive():
            self._active_worker.join(timeout=timeout)

    def get_recent(self, limit: int = 50, session_id: str | None = None) -> list[Episode]:
        """Retrieves recent episodes ordered chronologically descending."""
        self.flush()
        query = "SELECT * FROM episodes"
        params = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self.db.get_connection()
        rows = conn.execute(query, params).fetchall()

        return [self._row_to_episode(r) for r in rows]

    def get_unconsolidated(self, limit: int = 200, session_id: str | None = None) -> list[Episode]:
        """Retrieves raw episodes that have not yet been processed by a sleep consolidation cycle."""
        self.flush()
        query = "SELECT * FROM episodes WHERE consolidated_at IS NULL"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        conn = self.db.get_connection()
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def mark_consolidated(self, episode_ids: list[str], consolidated_at: float) -> int:
        """Marks a batch of episodes as consolidated."""
        if not episode_ids:
            return 0
        self.flush()
        conn = self.db.get_connection()
        with self.db.write_transaction(conn):
            placeholders = ",".join(["?"] * len(episode_ids))
            cur = conn.execute(
                f"UPDATE episodes SET consolidated_at = ? WHERE id IN ({placeholders})",
                [consolidated_at, *episode_ids]
            )
            return cur.rowcount

    def prune_older_than(self, retention_seconds: float, only_consolidated: bool = True) -> int:
        """Prunes historical episodes older than retention threshold to prevent disk bloat."""
        cutoff_time = time.time() - retention_seconds
        self.flush()
        conn = self.db.get_connection()
        with self.db.write_transaction(conn):
            if only_consolidated:
                cur = conn.execute(
                    "DELETE FROM episodes WHERE timestamp < ? AND consolidated_at IS NOT NULL",
                    (cutoff_time,)
                )
            else:
                cur = conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff_time,))
            return cur.rowcount

    def count(self, session_id: str | None = None) -> int:
        """Returns total episode count."""
        self.flush()
        query = "SELECT COUNT(*) FROM episodes"
        params = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        conn = self.db.get_connection()
        return conn.execute(query, params).fetchone()[0]

    @staticmethod
    def _row_to_episode(r) -> Episode:
        consolidated = r["consolidated_at"] if "consolidated_at" in r else None
        return Episode(
            id=r["id"],
            timestamp=r["timestamp"],
            session_id=r["session_id"],
            context=r["context"] or "",
            action=r["action"],
            result=r["result"] or "",
            status=ActionStatus(r["status"]),
            latency_ms=r["latency_ms"],
            metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            consolidated_at=consolidated
        )
