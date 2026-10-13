"""
store.py - Typed access to the memory graph: episodes, entities, predicates, facts,
summaries and procedures, each kept in sync with its vector and lexical index.
A single re-entrant lock serializes access so the proxy's request path and background
sleep can share one store.
"""

import json
import threading
import time
from contextlib import contextmanager

import numpy as np

from smruti.config import SmrutiConfig
from smruti.embed import Embedder, FastEmbedder
from smruti.store import index
from smruti.store.db import connect, init_schema
from smruti.store.records import Episode, from_row

USER_ENTITY = "user"


def _now(now: float | None) -> float:
    return time.time() if now is None else now


def _placeholders(ids) -> str:
    return ",".join("?" * len(ids))


class Store:
    def __init__(self, config: SmrutiConfig, embedder: Embedder | None = None):
        self.config = config
        self.embedder = embedder or FastEmbedder(config.embedding_model, config.embedding_dim)
        self.lock = threading.RLock()
        self.conn = connect(config.db_path)
        init_schema(self.conn, self.embedder.dim)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    @contextmanager
    def tx(self):
        """Write transaction; nested calls join the outer transaction."""
        with self.lock:
            if self.conn.in_transaction:
                yield self.conn
                return
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def _rows(self, sql: str, params=()) -> list:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def _by_ids(self, cls, table: str, ids) -> list:
        ids = list(ids)
        if not ids:
            return []
        rows = {r["id"]: r for r in self._rows(f"SELECT * FROM {table} WHERE id IN ({_placeholders(ids)})", ids)}
        return [from_row(cls, rows[i]) for i in ids if i in rows]

    # --- Embeddings and index search -------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    def vec_search(self, kind: str, space: str, vec, k: int) -> list[tuple[int, float]]:
        with self.lock:
            return index.vec_search(self.conn, kind, space, vec, k)

    def text_search(self, kind: str, space: str, text: str, k: int) -> list[tuple[int, float]]:
        with self.lock:
            return index.text_search(self.conn, kind, space, text, k)

    # --- Episodes ----------------------------------------------------------------------

    def add_episode(self, space, role, text, outcome, novelty, vec, now=None) -> int:
        with self.tx() as c:
            episode_id = c.execute(
                "INSERT INTO episodes (space, role, text, outcome, novelty, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (space, role, text, outcome, novelty, _now(now)),
            ).lastrowid
            index.vec_put(c, "episodes", episode_id, space, vec)
            index.text_put(c, "episodes", episode_id, space, text)
        return episode_id

    def bump_seen(self, episode_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE episodes SET seen_count = seen_count + 1 WHERE id = ?", (episode_id,))

    def episodes(self, ids) -> list[Episode]:
        return self._by_ids(Episode, "episodes", ids)

    def pending_episodes(self, space: str, limit: int) -> list[Episode]:
        rows = self._rows(
            "SELECT * FROM episodes WHERE space = ? AND consolidated_at IS NULL ORDER BY created_at, id LIMIT ?",
            (space, limit),
        )
        return [from_row(Episode, r) for r in rows]

    def pending_stats(self) -> list[tuple[str, int, float]]:
        """(space, pending count, newest pending created_at) for every space with pending episodes."""
        rows = self._rows(
            "SELECT space, COUNT(*) AS pending, MAX(created_at) AS newest FROM episodes "
            "WHERE consolidated_at IS NULL GROUP BY space ORDER BY space"
        )
        return [(r["space"], r["pending"], r["newest"]) for r in rows]

    def mark_consolidated(self, ids, now: float) -> None:
        with self.tx() as c:
            c.executemany("UPDATE episodes SET consolidated_at = ? WHERE id = ?", [(now, i) for i in ids])

    def record_sleep_failure(self, ids, error: str, max_attempts: int, now: float) -> None:
        """Counts a failed sleep attempt; episodes that keep failing are retired so sleep cannot loop."""
        with self.tx() as c:
            for episode_id in ids:
                c.execute(
                    "UPDATE episodes SET sleep_attempts = sleep_attempts + 1, sleep_error = ? WHERE id = ?",
                    (error[:500], episode_id),
                )
                c.execute(
                    "UPDATE episodes SET consolidated_at = ? WHERE id = ? AND sleep_attempts >= ?",
                    (now, episode_id, max_attempts),
                )

    def prune_episodes(self, before: float) -> int:
        with self.tx() as c:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM episodes WHERE consolidated_at IS NOT NULL AND consolidated_at < ?", (before,)
                ).fetchall()
            ]
            for episode_id in ids:
                c.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
                index.vec_drop(c, "episodes", episode_id)
                index.text_drop(c, "episodes", episode_id)
        return len(ids)

    # --- Novelty statistics ------------------------------------------------------------

    def novelty_scores(self, space: str) -> list[float]:
        rows = self._rows("SELECT scores FROM novelty_stats WHERE space = ?", (space,))
        return json.loads(rows[0]["scores"]) if rows else []

    def push_novelty(self, space: str, score: float, window: int) -> None:
        with self.tx() as c:
            scores = (self.novelty_scores(space) + [float(score)])[-window:]
            c.execute(
                "INSERT INTO novelty_stats (space, scores) VALUES (?, ?) "
                "ON CONFLICT(space) DO UPDATE SET scores = excluded.scores",
                (space, json.dumps(scores)),
            )
