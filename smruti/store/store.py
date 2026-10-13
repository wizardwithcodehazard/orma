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
from smruti.store.records import Entity, Episode, Fact, Predicate, from_row

USER_ENTITY = "user"

_FACT_SELECT = """
SELECT f.*, s.name AS subject, p.name AS predicate, COALESCE(o.name, f.object_value) AS object
FROM facts f
JOIN entities s ON s.id = f.subject_id
JOIN predicates p ON p.id = f.predicate_id
LEFT JOIN entities o ON o.id = f.object_id
"""

# Breadth-first walk over current entity-to-entity facts; each fact gets the hop distance
# of the nearest walked entity it touches.
_NEIGHBORHOOD = """
WITH RECURSIVE walk(entity_id, hops) AS (
    SELECT value, 0 FROM json_each(?)
    UNION
    SELECT CASE WHEN f.subject_id = w.entity_id THEN f.object_id ELSE f.subject_id END, w.hops + 1
    FROM walk w JOIN facts f ON f.subject_id = w.entity_id OR f.object_id = w.entity_id
    WHERE f.space = ? AND f.valid_to IS NULL AND f.object_id IS NOT NULL AND w.hops + 1 < ?
)
SELECT f.id AS id, MIN(w.hops) AS hops
FROM walk w JOIN facts f ON f.subject_id = w.entity_id OR f.object_id = w.entity_id
WHERE f.space = ? AND f.valid_to IS NULL
GROUP BY f.id
"""


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

    # --- Entities ----------------------------------------------------------------------

    def user_entity(self, space: str, now=None) -> int:
        """The entity for the person chatting; created on first use."""
        with self.lock:
            rows = self._rows("SELECT id FROM entities WHERE space = ? AND name = ?", (space, USER_ENTITY))
            return rows[0]["id"] if rows else self.add_entity(space, USER_ENTITY, "person", now=now)

    def add_entity(self, space: str, name: str, type_: str, vec=None, now=None) -> int:
        vec = self.embed(name) if vec is None else vec
        with self.tx() as c:
            entity_id = c.execute(
                "INSERT INTO entities (space, name, type, last_seen) VALUES (?, ?, ?, ?)",
                (space, name, type_, _now(now)),
            ).lastrowid
            index.vec_put(c, "entities", entity_id, space, vec)
            index.text_put(c, "entities", entity_id, space, name)
        return entity_id

    def touch_entity(self, entity_id: int, now=None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE entities SET mention_count = mention_count + 1, last_seen = ? WHERE id = ?",
                (_now(now), entity_id),
            )

    def entity(self, entity_id: int) -> Entity | None:
        found = self._by_ids(Entity, "entities", [entity_id])
        return found[0] if found else None

    def find_entity(self, space: str, name: str) -> Entity | None:
        rows = self._rows(
            "SELECT * FROM entities WHERE space = ? AND lower(name) = lower(?) ORDER BY id LIMIT 1",
            (space, name.strip()),
        )
        return from_row(Entity, rows[0]) if rows else None

    def nearest_entities(self, space: str, vec, k: int) -> list[tuple[Entity, float]]:
        hits = self.vec_search("entities", space, vec, k)
        found = {e.id: e for e in self._by_ids(Entity, "entities", [i for i, _ in hits])}
        return [(found[i], sim) for i, sim in hits if i in found]

    # --- Predicates --------------------------------------------------------------------

    def add_predicate(self, space: str, name: str, cardinality: str, vec=None) -> int:
        vec = self.embed(name.replace("_", " ")) if vec is None else vec
        with self.tx() as c:
            predicate_id = c.execute(
                "INSERT INTO predicates (space, name, cardinality) VALUES (?, ?, ?)", (space, name, cardinality)
            ).lastrowid
            index.vec_put(c, "predicates", predicate_id, space, vec)
        return predicate_id

    def predicate(self, predicate_id: int) -> Predicate | None:
        found = self._by_ids(Predicate, "predicates", [predicate_id])
        return found[0] if found else None

    def predicates(self, space: str) -> list[Predicate]:
        return [from_row(Predicate, r) for r in self._rows("SELECT * FROM predicates WHERE space = ? ORDER BY id", (space,))]

    def nearest_predicate(self, space: str, vec) -> tuple[Predicate, float] | None:
        hits = self.vec_search("predicates", space, vec, 1)
        return (self.predicate(hits[0][0]), hits[0][1]) if hits else None

    # --- Facts -------------------------------------------------------------------------

    def add_fact(self, space, subject_id, predicate_id, object_id, object_value, valid_from, source_episode_ids, now=None) -> int:
        """Inserts a current fact with one observation of support (confidence 1/2)."""
        with self.tx() as c:
            fact_id = c.execute(
                "INSERT INTO facts (space, subject_id, predicate_id, object_id, object_value, confidence, support, "
                "valid_from, last_used, source_episode_ids) VALUES (?, ?, ?, ?, ?, 0.5, 1, ?, ?, ?)",
                (space, subject_id, predicate_id, object_id, object_value, valid_from, _now(now),
                 json.dumps(sorted(set(source_episode_ids)))),
            ).lastrowid
            text = self.facts([fact_id])[0].text
            index.vec_put(c, "facts", fact_id, space, self.embed(text))
            index.text_put(c, "facts", fact_id, space, text)
        return fact_id

    def facts(self, ids) -> list[Fact]:
        ids = list(ids)
        if not ids:
            return []
        rows = {r["id"]: r for r in self._rows(_FACT_SELECT + f" WHERE f.id IN ({_placeholders(ids)})", ids)}
        return [Fact.from_row(rows[i]) for i in ids if i in rows]

    def current_facts(self, space: str, subject_id=None, predicate_id=None) -> list[Fact]:
        sql, params = _FACT_SELECT + " WHERE f.space = ? AND f.valid_to IS NULL", [space]
        if subject_id is not None:
            sql += " AND f.subject_id = ?"
            params.append(subject_id)
        if predicate_id is not None:
            sql += " AND f.predicate_id = ?"
            params.append(predicate_id)
        return [Fact.from_row(r) for r in self._rows(sql + " ORDER BY f.id", params)]

    def fact_history(self, space: str, subject_id=None) -> list[Fact]:
        """All facts including closed ones, oldest first: the timeline for 'when' questions."""
        sql, params = _FACT_SELECT + " WHERE f.space = ?", [space]
        if subject_id is not None:
            sql += " AND f.subject_id = ?"
            params.append(subject_id)
        return [Fact.from_row(r) for r in self._rows(sql + " ORDER BY f.valid_from, f.id", params)]

    def reinforce_fact(self, fact_id: int, source_episode_ids, now=None) -> None:
        """Another independent observation: support + 1, confidence = support / (support + 1)."""
        with self.tx() as c:
            row = c.execute("SELECT support, source_episode_ids FROM facts WHERE id = ?", (fact_id,)).fetchone()
            support = row["support"] + 1
            sources = sorted(set(json.loads(row["source_episode_ids"])) | set(source_episode_ids))
            c.execute(
                "UPDATE facts SET support = ?, confidence = ?, source_episode_ids = ?, last_used = ? WHERE id = ?",
                (support, support / (support + 1), json.dumps(sources), _now(now), fact_id),
            )

    def close_fact(self, fact_id: int, valid_to: float) -> None:
        """Ends a fact's validity: it stays in the timeline but leaves the search indexes."""
        with self.tx() as c:
            c.execute("UPDATE facts SET valid_to = ? WHERE id = ? AND valid_to IS NULL", (valid_to, fact_id))
            index.vec_drop(c, "facts", fact_id)
            index.text_drop(c, "facts", fact_id)

    def neighborhood(self, space: str, entity_ids, max_hops: int) -> dict[int, int]:
        """Current facts within max_hops edges of the given entities, as {fact id: hop distance}."""
        entity_ids = list(entity_ids)
        if not entity_ids or max_hops < 1:
            return {}
        rows = self._rows(_NEIGHBORHOOD, (json.dumps(entity_ids), space, max_hops, space))
        return {r["id"]: r["hops"] for r in rows}
