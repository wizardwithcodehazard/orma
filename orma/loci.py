"""
loci.py - Method of Loci (Topological Memory Palace) for Orma.
=============================================================
Translates spatial mnemonics (Ron White Method of Loci / ancient Greek mnemonics)
into a dense-vector graph-anchored cognitive retrieval mechanism for multi-session dialogues.

Concepts:
- Locus (Room/Checkpoint): An ordered temporal/thematic waypoint along the conversation route,
  anchored by a dense semantic embedding vector in multilingual metric space.
- Pegging: Semantic states and episodic turns are anchored directly to specific Loci.
- Palace Walk: When a query arrives, retrieval computes cosine similarity against Loci vectors
  to identify entrance checkpoints, then sequentially "walks" the adjacent precursor and
  successor rooms (ordinal - radius to ordinal + radius), reconstructing the chronological
  narrative chain for multi-hop reasoning with zero language-dependent heuristics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orma.store.records import Sentence, State
    from orma.store.store import Store


def _to_blob(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(b: bytes | None) -> np.ndarray | None:
    if not b:
        return None
    return np.frombuffer(b, dtype=np.float32)


@dataclass
class Locus:
    id: int
    space: str
    ordinal: int
    label: str
    gist: str = ""
    vec: np.ndarray | None = None
    occurred_at: float | None = None
    created_at: float = 0.0


@dataclass
class PalaceWalkResult:
    query: str
    entry_loci: list[Locus] = field(default_factory=list)
    walked_loci: list[Locus] = field(default_factory=list)
    states: list[State] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    narrative_text: str = ""


class Palace:
    """Manages the topological Memory Palace checkpoints and narrative walks."""

    def __init__(self, store: Store):
        self.store = store

    def add_locus(
        self,
        space: str,
        label: str,
        gist: str = "",
        occurred_at: float | None = None,
        now: float | None = None,
    ) -> Locus:
        """Creates a new sequential checkpoint (Room) in the space's Memory Palace."""
        now = time.time() if now is None else now
        anchor_text = f"{label}: {gist}" if gist else label
        vec = self.store.embed(anchor_text)
        vec_blob = _to_blob(vec)

        with self.store.tx() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM loci WHERE space = ?",
                (space,),
            ).fetchone()
            ordinal = int(row[0]) if row else 1

            locus_id = c.execute(
                "INSERT INTO loci (space, ordinal, label, gist, vec, occurred_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (space, ordinal, label, gist, vec_blob, occurred_at, now),
            ).lastrowid

        return Locus(
            id=locus_id,
            space=space,
            ordinal=ordinal,
            label=label,
            gist=gist,
            vec=vec,
            occurred_at=occurred_at,
            created_at=now,
        )

    def current_locus(self, space: str) -> Locus | None:
        """Returns the most recent active locus (Room) in the palace."""
        row = self.store.conn.execute(
            "SELECT id, space, ordinal, label, gist, vec, occurred_at, created_at "
            "FROM loci WHERE space = ? ORDER BY ordinal DESC LIMIT 1",
            (space,),
        ).fetchone()
        if not row:
            return None
        return Locus(
            id=row["id"],
            space=row["space"],
            ordinal=row["ordinal"],
            label=row["label"],
            gist=row["gist"],
            vec=_from_blob(row["vec"]),
            occurred_at=row["occurred_at"],
            created_at=row["created_at"],
        )

    def peg_episode(self, locus_id: int, episode_id: int) -> None:
        """Pegs (anchors) an episodic turn to a specific Palace Locus."""
        with self.store.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO locus_episodes (locus_id, episode_id) VALUES (?, ?)",
                (locus_id, episode_id),
            )

    def peg_state(self, locus_id: int, state_id: int) -> None:
        """Pegs (anchors) a consolidated semantic state to a specific Palace Locus."""
        with self.store.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO locus_states (locus_id, state_id) VALUES (?, ?)",
                (locus_id, state_id),
            )

    def list_loci(self, space: str) -> list[Locus]:
        """Returns all palace checkpoints in sequential walk order."""
        rows = self.store.conn.execute(
            "SELECT id, space, ordinal, label, gist, vec, occurred_at, created_at "
            "FROM loci WHERE space = ? ORDER BY ordinal ASC",
            (space,),
        ).fetchall()
        return [
            Locus(
                id=r["id"],
                space=r["space"],
                ordinal=r["ordinal"],
                label=r["label"],
                gist=r["gist"],
                vec=_from_blob(r["vec"]),
                occurred_at=r["occurred_at"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def walk(
        self,
        space: str,
        query: str,
        radius: int = 1,
        max_states: int = 15,
        max_sentences: int = 10,
    ) -> PalaceWalkResult:
        """Performs a Method of Loci Palace Walk:
        1. Identifies entry loci via multilingual dense vector cosine similarity against Loci anchors.
        2. Traverses adjacent loci within topological radius (ordinal - radius to ordinal + radius).
        3. Reconstructs chronologically ordered states and grounding sentences.
        """
        all_loci = self.list_loci(space)
        if not all_loci:
            return PalaceWalkResult(query=query)

        q_vec = self.store.embed(query)
        q_norm = float(np.linalg.norm(q_vec)) if q_vec is not None else 0.0

        # Score loci via cosine similarity in dense vector metric space
        scored_loci: list[tuple[float, Locus]] = []
        for loc in all_loci:
            sim = 0.0
            if loc.vec is not None and q_norm > 1e-9:
                loc_norm = float(np.linalg.norm(loc.vec))
                if loc_norm > 1e-9:
                    sim = float(np.dot(q_vec, loc.vec) / (q_norm * loc_norm))
            scored_loci.append((sim, loc))

        scored_loci.sort(key=lambda x: -x[0])

        # Primary entry checkpoint is the highest semantic similarity anchor above noise floor
        if not query.strip():
            entry = [all_loci[-1]]
        elif scored_loci and scored_loci[0][0] >= 0.52:
            top_sim = scored_loci[0][0]
            if radius == 0:
                entry = [scored_loci[0][1]]
            else:
                entry = [loc for sim, loc in scored_loci if sim >= top_sim * 0.92][:2]
        else:
            entry = [all_loci[-1]]

        # Expand topological route (Palace Walk)
        walked_ordinals: set[int] = set()
        for ent in entry:
            start_ord = max(1, ent.ordinal - radius)
            end_ord = ent.ordinal + radius
            for o in range(start_ord, end_ord + 1):
                walked_ordinals.add(o)

        walked = [loc for loc in all_loci if loc.ordinal in walked_ordinals]
        walked.sort(key=lambda x: x.ordinal)

        walked_locus_ids = [loc.id for loc in walked]
        if not walked_locus_ids:
            return PalaceWalkResult(query=query, entry_loci=entry)

        # Retrieve pegged states in topological order
        placeholders = ",".join("?" * len(walked_locus_ids))
        state_rows = self.store.conn.execute(
            f"SELECT DISTINCT s.id, ls.locus_id FROM states s "
            f"JOIN locus_states ls ON ls.state_id = s.id "
            f"WHERE ls.locus_id IN ({placeholders}) AND s.valid_to IS NULL "
            f"ORDER BY s.valid_from ASC LIMIT ?",
            (*walked_locus_ids, max_states),
        ).fetchall()
        state_ids = [r[0] for r in state_rows]
        locus_to_state_ids: dict[int, list[int]] = {}
        for r in state_rows:
            locus_to_state_ids.setdefault(r[1], []).append(r[0])
        states = self.store.states(state_ids)
        states_by_id = {st.id: st for st in states}

        # Retrieve pegged sentences in topological order
        sentence_rows = self.store.conn.execute(
            f"SELECT DISTINCT s.id FROM sentences s "
            f"JOIN locus_episodes le ON le.episode_id = s.episode_id "
            f"WHERE le.locus_id IN ({placeholders}) "
            f"ORDER BY s.created_at ASC, s.position ASC LIMIT ?",
            (*walked_locus_ids, max_sentences),
        ).fetchall()
        sentence_ids = [r[0] for r in sentence_rows]
        sentences = self.store.sentences(sentence_ids)

        # Assemble coherent narrative block
        lines = []
        for loc in walked:
            lines.append(f"[{loc.label}]: {loc.gist or 'Checkpoint'}")
            st_ids = locus_to_state_ids.get(loc.id, [])
            for sid in st_ids[:4]:
                if sid in states_by_id:
                    lines.append(f"  - {states_by_id[sid].value}")

        narrative = "\n".join(lines)
        return PalaceWalkResult(
            query=query,
            entry_loci=entry,
            walked_loci=walked,
            states=states,
            sentences=sentences,
            narrative_text=narrative,
        )
