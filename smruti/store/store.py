"""
store.py - Typed access to the memory: kept records (episodes) and their sentences, entities, states and
events, each kept in sync with its vector and lexical index. A single re-entrant lock serializes access
so the proxy's request path and background sleep can share one store.
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
from smruti.store.records import Attribute, Entity, Episode, Event, Sentence, State, from_row, state_text

USER_ENTITY = "user"

_STATE_SELECT = """
SELECT st.*, e.name AS subject, a.name AS attribute
FROM states st
JOIN entities e ON e.id = st.subject_id
JOIN attributes a ON a.id = st.attribute_id
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

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(list(texts))

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

    # --- Episodes (kept records) -------------------------------------------------------

    def add_episode(self, space, role, text, outcome, novelty, vec, now=None, source="chat", speaker="") -> int:
        with self.tx() as c:
            episode_id = c.execute(
                "INSERT INTO episodes (space, role, text, outcome, novelty, created_at, source, speaker) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (space, role, text, outcome, novelty, _now(now), source, speaker or role),
            ).lastrowid
            index.vec_put(c, "episodes", episode_id, space, vec)
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

    def episodes_without_sentences(self, space: str) -> list[Episode]:
        """Records kept before sentences existed (a version 1 database)."""
        rows = self._rows("SELECT * FROM episodes e WHERE space = ? AND NOT EXISTS "
                          "(SELECT 1 FROM sentences s WHERE s.episode_id = e.id) ORDER BY created_at, id", (space,))
        return [from_row(Episode, r) for r in rows]

    # --- Sentences ---------------------------------------------------------------------

    def add_sentences(self, space: str, episode: Episode, texts: list[str], vecs) -> list[int]:
        ids = []
        with self.tx() as c:
            for position, (text, vec) in enumerate(zip(texts, vecs)):
                sentence_id = c.execute(
                    "INSERT INTO sentences (space, episode_id, position, role, speaker, text, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (space, episode.id, position, episode.role, episode.speaker or episode.role, text,
                     episode.created_at),
                ).lastrowid
                index.vec_put(c, "sentences", sentence_id, space, vec)
                index.text_put(c, "sentences", sentence_id, space, text)
                ids.append(sentence_id)
        return ids

    def sentences(self, ids) -> list[Sentence]:
        return self._by_ids(Sentence, "sentences", ids)

    def episode_sentences(self, episode_ids) -> list[Sentence]:
        """The sentences of the given episodes, in the order they were said."""
        ids = list(episode_ids)
        if not ids:
            return []
        rows = self._rows(f"SELECT * FROM sentences WHERE episode_id IN ({_placeholders(ids)}) "
                          "ORDER BY created_at, episode_id, position", ids)
        return [from_row(Sentence, r) for r in rows]

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
        """The entity for the person this memory belongs to; created on first use."""
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

    def entities_named_in(self, space: str, text: str, k: int = 8) -> list[Entity]:
        """Known entities whose whole name occurs in text."""
        hits = [hit.id for hit in self.text_search("entities", space, text, k) if hit.row_coverage >= 1.0]
        return self._by_ids(Entity, "entities", hits)

    # --- Attributes --------------------------------------------------------------------

    def add_attribute(self, space: str, name: str, vec=None) -> int:
        vec = self.embed(name.replace("_", " ")) if vec is None else vec
        with self.tx() as c:
            attribute_id = c.execute("INSERT INTO attributes (space, name) VALUES (?, ?)", (space, name)).lastrowid
            index.vec_put(c, "attributes", attribute_id, space, vec)
        return attribute_id

    def attribute(self, attribute_id: int) -> Attribute | None:
        found = self._by_ids(Attribute, "attributes", [attribute_id])
        return found[0] if found else None

    def find_attribute(self, space: str, name: str) -> Attribute | None:
        rows = self._rows("SELECT * FROM attributes WHERE space = ? AND name = ?", (space, name))
        return from_row(Attribute, rows[0]) if rows else None

    def nearest_attributes(self, space: str, vec, k: int) -> list[tuple[Attribute, float]]:
        with self.lock:
            hits = self.vec_search("attributes", space, vec, k)
            found = {a.id: a for a in self._by_ids(Attribute, "attributes", [i for i, _ in hits])}
            return [(found[i], sim) for i, sim in hits if i in found]

    # --- States ------------------------------------------------------------------------

    def add_state(self, space: str, subject_id: int, attribute_id: int, value: str, valid_from: float,
                  sentence_id: int, said_at: float, stated: bool = False, valid_to: float | None = None,
                  vec=None) -> int:
        """A state stated by sentence_id (said at said_at); current unless valid_to is given. Pass a precomputed
        vec of its text to keep inference out of the caller's transaction."""
        text = state_text(self.entity(subject_id).name, self.attribute(attribute_id).name, value)
        vec = self.embed(text) if vec is None else vec
        with self.tx() as c:
            state_id = c.execute(
                "INSERT INTO states (space, subject_id, attribute_id, value, valid_from, valid_from_stated, valid_to, "
                "sentence_id, evidence, evidence_times) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (space, subject_id, attribute_id, value, valid_from, int(stated), valid_to, sentence_id,
                 json.dumps([sentence_id]), json.dumps([said_at])),
            ).lastrowid
            if valid_to is None:  # past states stay out of the search indexes
                index.vec_put(c, "states", state_id, space, vec)
                index.text_put(c, "states", state_id, space, text)
        return state_id

    def states(self, ids) -> list[State]:
        ids = list(ids)
        if not ids:
            return []
        rows = {r["id"]: r for r in self._rows(_STATE_SELECT + f" WHERE st.id IN ({_placeholders(ids)})", ids)}
        return [State.from_row(rows[i]) for i in ids if i in rows]

    def current_states(self, space: str, subject_id=None, attribute_id=None) -> list[State]:
        sql, params = _STATE_SELECT + " WHERE st.space = ? AND st.valid_to IS NULL", [space]
        if subject_id is not None:
            sql += " AND st.subject_id = ?"
            params.append(subject_id)
        if attribute_id is not None:
            sql += " AND st.attribute_id = ?"
            params.append(attribute_id)
        return [State.from_row(r) for r in self._rows(sql + " ORDER BY st.id", params)]

    def state_history(self, space: str, subject_id=None, attribute_id=None) -> list[State]:
        """All states including past ones, oldest first: the timeline for 'when' questions."""
        sql, params = _STATE_SELECT + " WHERE st.space = ?", [space]
        if subject_id is not None:
            sql += " AND st.subject_id = ?"
            params.append(subject_id)
        if attribute_id is not None:
            sql += " AND st.attribute_id = ?"
            params.append(attribute_id)
        return [State.from_row(r) for r in self._rows(sql + " ORDER BY st.valid_from, st.id", params)]

    def states_citing(self, sentence_ids) -> list[State]:
        """States (current or past) first stated by any of the given sentences."""
        ids = list(sentence_ids)
        if not ids:
            return []
        return [State.from_row(r) for r in
                self._rows(_STATE_SELECT + f" WHERE st.sentence_id IN ({_placeholders(ids)}) ORDER BY st.id", ids)]

    def reinforce_state(self, state_id: int, sentence_id: int, said_at: float) -> None:
        """Another sentence states the same value: one more use of this memory."""
        with self.tx() as c:
            row = c.execute("SELECT evidence, evidence_times FROM states WHERE id = ?", (state_id,)).fetchone()
            evidence, times = json.loads(row["evidence"]), json.loads(row["evidence_times"])
            if sentence_id not in evidence:
                c.execute("UPDATE states SET evidence = ?, evidence_times = ? WHERE id = ?",
                          (json.dumps(evidence + [sentence_id]), json.dumps(times + [said_at]), state_id))

    def close_state(self, state_id: int, valid_to: float, ended_by: int | None = None) -> None:
        """Ends a state: it stays in the timeline but leaves the search indexes."""
        with self.tx() as c:
            c.execute("UPDATE states SET valid_to = ?, ended_by = ? WHERE id = ? AND valid_to IS NULL",
                      (valid_to, ended_by, state_id))
            index.vec_drop(c, "states", state_id)
            index.text_drop(c, "states", state_id)

    # --- Events ------------------------------------------------------------------------

    def add_event(self, space: str, gist: str, occurred_at: float, occurred_stated: bool, participant_ids,
                  sentence_id: int, vec=None, now=None) -> int:
        """Inserts a dated event, stated by sentence_id, linked to its participant entities."""
        vec = self.embed(gist) if vec is None else vec
        with self.tx() as c:
            event_id = c.execute(
                "INSERT INTO events (space, gist, occurred_at, occurred_stated, created_at, sentence_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (space, gist, occurred_at, int(occurred_stated), _now(now), sentence_id),
            ).lastrowid
            c.executemany("INSERT OR IGNORE INTO event_participants (event_id, entity_id) VALUES (?, ?)",
                          [(event_id, entity_id) for entity_id in participant_ids])
            index.vec_put(c, "events", event_id, space, vec)
            index.text_put(c, "events", event_id, space, gist)
        return event_id

    def events(self, ids) -> list[Event]:
        return self._by_ids(Event, "events", ids)

    def space_events(self, space: str) -> list[Event]:
        """Every event of a space, in the order they happened."""
        rows = self._rows("SELECT * FROM events WHERE space = ? ORDER BY occurred_at, id", (space,))
        return [from_row(Event, r) for r in rows]

    def events_citing(self, sentence_id: int) -> list[Event]:
        rows = self._rows("SELECT * FROM events WHERE sentence_id = ? ORDER BY id", (sentence_id,))
        return [from_row(Event, r) for r in rows]

    def event_participants(self, event_id: int) -> list[int]:
        return [r[0] for r in self._rows(
            "SELECT entity_id FROM event_participants WHERE event_id = ? ORDER BY entity_id", (event_id,))]

    # --- Rebuilding --------------------------------------------------------------------

    def clear_consolidation(self, space: str) -> None:
        """Forgets everything sleep built in the space (states, events, attributes, entities other than the user,
        aliases) and marks every kept record pending again, so it can be read anew. Records and sentences stay."""
        with self.tx() as c:
            for kind, table in (("states", "states"), ("events", "events"), ("attributes", "attributes")):
                for (row_id,) in c.execute(f"SELECT id FROM {table} WHERE space = ?", (space,)).fetchall():
                    index.vec_drop(c, kind, row_id)
                    if kind != "attributes":
                        index.text_drop(c, kind, row_id)
            c.execute("DELETE FROM event_participants WHERE event_id IN (SELECT id FROM events WHERE space = ?)",
                      (space,))
            c.execute("DELETE FROM events WHERE space = ?", (space,))
            c.execute("DELETE FROM states WHERE space = ?", (space,))
            c.execute("DELETE FROM attributes WHERE space = ?", (space,))
            c.execute("DELETE FROM entity_aliases WHERE space = ?", (space,))
            for (row_id,) in c.execute("SELECT id FROM entities WHERE space = ? AND name != ?",
                                       (space, USER_ENTITY)).fetchall():
                index.vec_drop(c, "entities", row_id)
                index.text_drop(c, "entities", row_id)
            c.execute("DELETE FROM entities WHERE space = ? AND name != ?", (space, USER_ENTITY))
            c.execute("UPDATE episodes SET consolidated_at = NULL, sleep_attempts = 0, sleep_error = NULL "
                      "WHERE space = ?", (space,))

    # --- Introspection -----------------------------------------------------------------

    def stats(self, space: str) -> dict[str, int]:
        counts = {
            "records": "SELECT COUNT(*) FROM episodes WHERE space = ?",
            "pending_records": "SELECT COUNT(*) FROM episodes WHERE space = ? AND consolidated_at IS NULL",
            "sentences": "SELECT COUNT(*) FROM sentences WHERE space = ?",
            "entities": "SELECT COUNT(*) FROM entities WHERE space = ?",
            "current_states": "SELECT COUNT(*) FROM states WHERE space = ? AND valid_to IS NULL",
            "past_states": "SELECT COUNT(*) FROM states WHERE space = ? AND valid_to IS NOT NULL",
            "events": "SELECT COUNT(*) FROM events WHERE space = ?",
        }
        return {name: self._rows(sql, (space,))[0][0] for name, sql in counts.items()}
