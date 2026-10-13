"""
store.py - Typed access to the memory graph: episodes, entities, predicates, facts,
events, summaries and procedures, each kept in sync with its vector and lexical index.
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
from smruti.store.records import (Entity, Episode, Event, Fact, Predicate, Procedure, Summary, fact_text,
                                  from_row)

USER_ENTITY = "user"

_FACT_SELECT = """
SELECT f.*, s.name AS subject, p.name AS predicate, COALESCE(o.name, f.object_value) AS object
FROM facts f
JOIN entities s ON s.id = f.subject_id
JOIN predicates p ON p.id = f.predicate_id
LEFT JOIN entities o ON o.id = f.object_id
"""

# Current facts touching any of the given entities, as subject or as object.
_INCIDENT = """
SELECT f.id, f.subject_id, f.object_id FROM facts f
WHERE f.space = ? AND f.valid_to IS NULL AND f.subject_id IN (SELECT value FROM json_each(?))
UNION
SELECT f.id, f.subject_id, f.object_id FROM facts f
WHERE f.space = ? AND f.valid_to IS NULL AND f.object_id IN (SELECT value FROM json_each(?))
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

    def text_search(self, kind: str, space: str, text: str, k: int) -> list[index.TextHit]:
        with self.lock:
            return index.text_search(self.conn, kind, space, text, k)

    def stems(self, texts: list[str]) -> list[set[str]]:
        """Each text's index terms, as the lexical index tokenizes and stems them."""
        with self.lock:
            return index.stems(self.conn, list(texts))

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

    def pending_by_ids(self, space: str, ids) -> list[Episode]:
        """The given episodes that are still unconsolidated in the space, in the order given."""
        ids = list(ids)
        if not ids:
            return []
        rows = {r["id"]: r for r in self._rows(
            f"SELECT * FROM episodes WHERE space = ? AND consolidated_at IS NULL AND id IN ({_placeholders(ids)})",
            [space, *ids])}
        return [from_row(Episode, rows[i]) for i in ids if i in rows]

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

    # --- Sleep lease -------------------------------------------------------------------

    def acquire_sleep_lease(self, space: str, holder: str, ttl: float, now: float) -> bool:
        """Takes (or renews) the space's sleep lease unless another holder's is still valid."""
        with self.tx() as c:
            row = c.execute("SELECT holder, expires_at FROM sleep_leases WHERE space = ?", (space,)).fetchone()
            if row is not None and row["holder"] != holder and row["expires_at"] > now:
                return False
            c.execute(
                "INSERT INTO sleep_leases (space, holder, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(space) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at",
                (space, holder, now + ttl),
            )
            return True

    def release_sleep_lease(self, space: str, holder: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM sleep_leases WHERE space = ? AND holder = ?", (space, holder))

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
        sql = "SELECT id FROM entities WHERE space = ? AND name = ?"
        rows = self._rows(sql, (space, USER_ENTITY))
        if rows:
            return rows[0]["id"]
        vec = self.embed(USER_ENTITY)  # inference stays outside the lock
        with self.lock:
            rows = self._rows(sql, (space, USER_ENTITY))
            return rows[0]["id"] if rows else self.add_entity(space, USER_ENTITY, "person", vec=vec, now=now)

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

    def add_alias(self, space: str, alias: str, entity_id: int) -> None:
        """Another name for an entity (stored lowercased); the first entity to claim an alias keeps it."""
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO entity_aliases (space, alias, entity_id) VALUES (?, ?, ?)",
                      (space, alias.strip().lower(), entity_id))

    def aliases(self, entity_id: int) -> list[str]:
        return [r["alias"] for r in self._rows(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias", (entity_id,))]

    def find_entity(self, space: str, name: str) -> Entity | None:
        aliased = self._rows(
            "SELECT e.* FROM entity_aliases a JOIN entities e ON e.id = a.entity_id WHERE a.space = ? AND a.alias = ?",
            (space, name.strip().lower()))
        if aliased:
            return from_row(Entity, aliased[0])
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

    def nearest_predicates(self, space: str, vec, k: int) -> list[tuple[Predicate, float]]:
        with self.lock:
            hits = self.vec_search("predicates", space, vec, k)
            found = {p.id: p for p in self._by_ids(Predicate, "predicates", [i for i, _ in hits])}
            return [(found[i], sim) for i, sim in hits if i in found]

    def nearest_predicate(self, space: str, vec) -> tuple[Predicate, float] | None:
        nearest = self.nearest_predicates(space, vec, 1)
        return nearest[0] if nearest else None

    # --- Facts -------------------------------------------------------------------------

    def add_fact(self, space, subject_id, predicate_id, object_id, object_value, valid_from, source_episode_ids, now=None,
                 valid_to=None, vec=None,
                 valid_from_stated: bool = False, support: int = 1) -> int:
        """Inserts a fact with support observations (one by default: confidence 1/2); current unless valid_to
        is given.
        Pass a precomputed vec (of the rendered text) to keep inference out of the caller's transaction."""
        subject = self.entity(subject_id).name
        predicate = self.predicate(predicate_id).name
        object_ = self.entity(object_id).name if object_id else object_value
        text = fact_text(subject, predicate, object_)
        vec = self.embed(text) if vec is None else vec  # inference stays outside the lock and transaction
        with self.tx() as c:
            fact_id = c.execute(
                "INSERT INTO facts (space, subject_id, predicate_id, object_id, object_value, confidence, support, "
                "valid_from, valid_to, last_used, source_episode_ids, valid_from_stated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (space, subject_id, predicate_id, object_id, object_value, support / (support + 1), support, valid_from, valid_to, _now(now),
                 json.dumps(sorted(set(source_episode_ids))), int(valid_from_stated)),
            ).lastrowid
            if valid_to is None:  # closed facts stay out of the search indexes
                index.vec_put(c, "facts", fact_id, space, vec)
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

    def reinforce_fact(self, fact_id: int, source_episode_ids, now=None, count: int = 1) -> None:
        """count more independent observations: support + count, confidence = support / (support + 1)."""
        with self.tx() as c:
            row = c.execute("SELECT support, source_episode_ids FROM facts WHERE id = ?", (fact_id,)).fetchone()
            support = row["support"] + count
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

    def incident_facts(self, space: str, entity_ids) -> list[tuple[int, int, int | None]]:
        """Current facts touching any of the given entities, as (fact id, subject id, object id or None).
        Every edge of each entity is returned, so the result also gives each entity's fan."""
        ids = json.dumps(list(entity_ids))
        return [(r[0], r[1], r[2]) for r in self._rows(_INCIDENT, (space, ids, space, ids))]

    # --- Events ------------------------------------------------------------------------

    def add_event(self, space: str, gist: str, occurred_at: float, occurred_stated: bool, participant_ids,
                  source_episode_ids, vec=None, now=None) -> int:
        """Inserts a dated event linked to its participant entities. Pass a precomputed vec (of the gist) to
        keep inference out of the caller's transaction."""
        vec = self.embed(gist) if vec is None else vec  # inference stays outside the lock and transaction
        with self.tx() as c:
            event_id = c.execute(
                "INSERT INTO events (space, gist, occurred_at, occurred_stated, created_at, source_episode_ids) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (space, gist, occurred_at, int(occurred_stated), _now(now), json.dumps(sorted(set(source_episode_ids)))),
            ).lastrowid
            c.executemany("INSERT OR IGNORE INTO event_participants (event_id, entity_id) VALUES (?, ?)",
                          [(event_id, entity_id) for entity_id in participant_ids])
            index.vec_put(c, "events", event_id, space, vec)
            index.text_put(c, "events", event_id, space, gist)
        return event_id

    def events(self, ids) -> list[Event]:
        return self._by_ids(Event, "events", ids)

    def batch_events(self, space: str, source_episode_ids) -> list[Event]:
        """Events recorded from exactly this batch of episodes, oldest first."""
        rows = self._rows("SELECT * FROM events WHERE space = ? AND source_episode_ids = ? ORDER BY id",
                          (space, json.dumps(sorted(set(source_episode_ids)))))
        return [from_row(Event, r) for r in rows]

    def event_participants(self, event_id: int) -> list[int]:
        return [r[0] for r in self._rows(
            "SELECT entity_id FROM event_participants WHERE event_id = ? ORDER BY entity_id", (event_id,))]

    def events_for_entities(self, space: str, entity_ids) -> list[tuple[int, int]]:
        """(event id, entity id) for every event any of the given entities took part in: the event edges of
        each entity, so the result also counts toward each entity's fan."""
        return [(r[0], r[1]) for r in self._rows(
            "SELECT p.event_id, p.entity_id FROM event_participants p JOIN events e ON e.id = p.event_id "
            "WHERE e.space = ? AND p.entity_id IN (SELECT value FROM json_each(?)) ORDER BY p.event_id, p.entity_id",
            (space, json.dumps(list(entity_ids))))]

    # --- Summaries ---------------------------------------------------------------------

    def upsert_summary(self, space: str, topic: str, text: str, tokens: int, now=None) -> int:
        indexed = f"{topic}: {text}"
        vec = self.embed(indexed)
        with self.tx() as c:
            c.execute(
                "INSERT INTO summaries (space, topic, text, tokens, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(space, topic) DO UPDATE SET text = excluded.text, tokens = excluded.tokens, "
                "updated_at = excluded.updated_at",
                (space, topic, text, tokens, _now(now)),
            )
            summary_id = c.execute(
                "SELECT id FROM summaries WHERE space = ? AND topic = ?", (space, topic)
            ).fetchone()["id"]
            index.vec_put(c, "summaries", summary_id, space, vec)
            index.text_put(c, "summaries", summary_id, space, indexed)
        return summary_id

    def summaries(self, ids) -> list[Summary]:
        return self._by_ids(Summary, "summaries", ids)

    def summary_topics(self, space: str) -> list[str]:
        return [r["topic"] for r in self._rows("SELECT topic FROM summaries WHERE space = ? ORDER BY topic", (space,))]

    def summary(self, space: str, topic: str) -> Summary | None:
        rows = self._rows("SELECT * FROM summaries WHERE space = ? AND topic = ?", (space, topic))
        return from_row(Summary, rows[0]) if rows else None

    # --- Procedures --------------------------------------------------------------------

    def add_procedure(self, space, trigger, action, error_class, works: bool, vec=None, now=None,
                      source_episode_ids=()) -> int:
        indexed = f"When {trigger}: {action}"
        vec = self.embed(indexed) if vec is None else vec
        now = _now(now)
        with self.tx() as c:
            procedure_id = c.execute(
                "INSERT INTO procedures (space, trigger, action, error_class, successes, failures, created_at, last_used, "
                "source_episode_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (space, trigger, action, error_class, int(works), int(not works), now, now,
                 json.dumps(sorted(set(source_episode_ids)))),
            ).lastrowid
            index.vec_put(c, "procedures", procedure_id, space, vec)
            index.text_put(c, "procedures", procedure_id, space, indexed)
        return procedure_id

    def add_procedure_evidence(self, procedure_id: int, works: bool, now=None, source_episode_ids=()) -> None:
        column = "successes" if works else "failures"
        with self.tx() as c:
            row = c.execute("SELECT source_episode_ids FROM procedures WHERE id = ?", (procedure_id,)).fetchone()
            sources = sorted(set(json.loads(row["source_episode_ids"])) | set(source_episode_ids))
            c.execute(
                f"UPDATE procedures SET {column} = {column} + 1, last_used = ?, source_episode_ids = ? WHERE id = ?",
                (_now(now), json.dumps(sources), procedure_id),
            )

    def procedures(self, ids) -> list[Procedure]:
        return self._by_ids(Procedure, "procedures", ids)

    def nearest_procedure(self, space: str, vec) -> tuple[Procedure, float] | None:
        with self.lock:
            hits = self.vec_search("procedures", space, vec, 1)
            found = self.procedures([hits[0][0]]) if hits else []
            return (found[0], hits[0][1]) if found else None

    def prune_procedures(self, unused_since: float, min_confidence: float) -> int:
        """Forgets procedures that are both stale and not confidently established."""
        with self.tx() as c:
            stale = [from_row(Procedure, r) for r in c.execute("SELECT * FROM procedures WHERE last_used < ?", (unused_since,))]
            doomed = [p.id for p in stale if p.confidence < min_confidence]
            for procedure_id in doomed:
                c.execute("DELETE FROM procedures WHERE id = ?", (procedure_id,))
                index.vec_drop(c, "procedures", procedure_id)
                index.text_drop(c, "procedures", procedure_id)
        return len(doomed)

    # --- Introspection -----------------------------------------------------------------

    def stats(self, space: str) -> dict[str, int]:
        counts = {
            "episodes": "SELECT COUNT(*) FROM episodes WHERE space = ?",
            "pending_episodes": "SELECT COUNT(*) FROM episodes WHERE space = ? AND consolidated_at IS NULL",
            "entities": "SELECT COUNT(*) FROM entities WHERE space = ?",
            "current_facts": "SELECT COUNT(*) FROM facts WHERE space = ? AND valid_to IS NULL",
            "closed_facts": "SELECT COUNT(*) FROM facts WHERE space = ? AND valid_to IS NOT NULL",
            "procedures": "SELECT COUNT(*) FROM procedures WHERE space = ?",
            "summaries": "SELECT COUNT(*) FROM summaries WHERE space = ?",
            "events": "SELECT COUNT(*) FROM events WHERE space = ?",
        }
        return {name: self._rows(sql, (space,))[0][0] for name, sql in counts.items()}
