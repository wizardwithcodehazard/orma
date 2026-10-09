"""
cortex.py - Tier 3A: Positive Knowledge Mesh with Biological Synaptic Decay (Ebbinghaus),
Neural Vector Embeddings, Semantic Deduplication, and Associative Spreading Activation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from smruti.config import smrutiConfig, get_config
from smruti.models import CorticalRule, RuleEdge, ValenceType
from smruti.storage.db import DatabaseManager, get_db
from smruti.storage.embeddings import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)

class Cortex:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        config: smrutiConfig | None = None,
        embedding_engine: EmbeddingEngine | None = None
    ):
        self.config = config or get_config()
        self.db = db or get_db(self.config)
        self.embeddings = embedding_engine or get_embedding_engine(self.config.embedding_model)
        self._vec_cache: dict[str, list[float]] = {}

    def add_rule(
        self,
        rule_text: str,
        category: str = "general",
        confidence: float = 1.0,
        base_strength: float = 1.0,
        source_episode_ids: list[str] | None = None
    ) -> CorticalRule:
        """
        Stores a verified positive heuristic or architectural invariant.
        Applies semantic deduplication: if a semantically equivalent rule already exists,
        reinforces the existing rule instead of creating an unwanted duplicate.
        Creates associative graph edges between closely related rules.
        """
        clean_text = rule_text.strip()
        category = category.strip()
        now = datetime.now(timezone.utc).timestamp()
        
        # 1. Compute embedding vector for semantic evaluation
        vec = self.embeddings.embed_text(clean_text)
        vec_json = json.dumps(vec)

        conn = self.db.get_connection()

        # 2. Semantic Deduplication Check
        existing_rows = conn.execute("SELECT * FROM cortical_rules").fetchall()
        for r in existing_rows:
            # Check exact text match or embedding similarity
            if r["rule_text"].strip().lower() == clean_text.lower():
                is_duplicate = True
                sim = 1.0
            else:
                existing_vec = json.loads(r["embedding"]) if r["embedding"] else None
                if existing_vec:
                    sim = EmbeddingEngine.cosine_similarity(vec, existing_vec)
                else:
                    sim = 0.0
                is_duplicate = (sim >= self.config.dedup_similarity_threshold)

            if is_duplicate:
                # Reinforce existing rule instead of duplicating
                existing_rule = self._row_to_rule(r)
                new_strength = min(1.0, existing_rule.base_strength + self.config.reinforcement_boost)
                new_confidence = max(existing_rule.confidence, confidence)
                new_count = existing_rule.access_count + 1

                # Merge source episode ids
                prev_sources = existing_rule.source_episode_ids or []
                if source_episode_ids:
                    merged_sources = list(set(prev_sources + source_episode_ids))
                else:
                    merged_sources = prev_sources

                with conn:
                    conn.execute(
                        """
                        UPDATE cortical_rules
                        SET access_count = ?,
                            last_accessed_at = ?,
                            base_strength = ?,
                            confidence = ?,
                            source_episode_ids = ?
                        WHERE id = ?
                        """,
                        (new_count, now, new_strength, new_confidence, json.dumps(merged_sources), existing_rule.id)
                    )

                existing_rule.access_count = new_count
                existing_rule.last_accessed_at = now
                existing_rule.base_strength = new_strength
                existing_rule.confidence = new_confidence
                existing_rule.source_episode_ids = merged_sources
                logger.info(
                    "Reinforced existing cortical rule '%s' (similarity: %.3f)",
                    existing_rule.id, sim
                )
                return existing_rule

        # 3. Create and insert new rule
        rule = CorticalRule(
            rule_text=clean_text,
            category=category,
            confidence=confidence,
            base_strength=base_strength,
            valence=ValenceType.POSITIVE,
            embedding=vec,
            source_episode_ids=source_episode_ids or []
        )

        with conn:
            conn.execute(
                """
                INSERT INTO cortical_rules (
                    id, rule_text, category, confidence, access_count,
                    created_at, last_accessed_at, base_strength, valence,
                    embedding, source_episode_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.id,
                    rule.rule_text,
                    rule.category,
                    rule.confidence,
                    rule.access_count,
                    rule.created_at,
                    rule.last_accessed_at,
                    rule.base_strength,
                    rule.valence.value,
                    vec_json,
                    json.dumps(rule.source_episode_ids)
                )
            )

        # 4. Connect associative edges to other semantically related rules
        for r in existing_rows:
            other_id = r["id"]
            if other_id == rule.id:
                continue
            other_vec = json.loads(r["embedding"]) if r["embedding"] else None
            if other_vec:
                sim = EmbeddingEngine.cosine_similarity(vec, other_vec)
                if sim >= self.config.associative_edge_threshold:
                    self._create_edge(rule.id, other_id, sim, now)

        return rule

    def recall_rules(
        self,
        query: str = "",
        category: str | None = None,
        limit: int = 5,
        current_time: float | None = None,
        reinforce: bool = True
    ) -> list[tuple[CorticalRule, float]]:
        """
        Retrieves top rules weighted by biological effective strength (Ebbinghaus retention),
        dense vector semantic similarity, and associative spreading activation across graph edges.
        If reinforce=True, reinforces recalled rules via Hebbian potentiation.
        """
        now = current_time or datetime.now(timezone.utc).timestamp()
        conn = self.db.get_connection()

        sql = "SELECT * FROM cortical_rules"
        params = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []

        # 1. Compute query vector if query provided
        has_query = bool(query and query.strip())
        query_vec = self.embeddings.embed_text(query) if has_query else None
        query_terms = [t.lower() for t in query.split() if len(t) > 2] if has_query else []

        rules_by_id: dict[str, CorticalRule] = {}
        base_scores: dict[str, float] = {}

        # Vectorized batch similarity calculation
        sims_by_id: dict[str, float] = {}
        if has_query and query_vec:
            vec_list: list[list[float]] = []
            id_list: list[str] = []
            for r in rows:
                rid = r["id"]
                if rid in self._vec_cache:
                    v = self._vec_cache[rid]
                elif r["embedding"]:
                    v = json.loads(r["embedding"])
                    self._vec_cache[rid] = v
                else:
                    v = None

                if v:
                    vec_list.append(v)
                    id_list.append(rid)

            if vec_list:
                matrix = np.asarray(vec_list, dtype=np.float32)
                sims = EmbeddingEngine.batch_cosine_similarity(query_vec, matrix)
                for rid, sim in zip(id_list, sims):
                    sims_by_id[rid] = float(sim)

        for r in rows:
            rule = self._row_to_rule(r)
            rules_by_id[rule.id] = rule

            effective_strength = rule.calculate_effective_strength(
                decay_rate=self.config.default_decay_rate,
                current_time=now
            )

            relevance = 1.0
            if has_query and query_vec:
                sem_sim = sims_by_id.get(rule.id, 0.0)
                sem_score = max(0.0, sem_sim)

                # Lexical overlap bonus
                lex_matches = sum(1 for t in query_terms if t in rule.rule_text.lower() or t in rule.category.lower())
                lex_bonus = min(0.4, lex_matches * 0.15)

                # Combined neural + symbolic relevance multiplier
                relevance = 1.0 + (sem_score * 2.0) + lex_bonus

            base_scores[rule.id] = effective_strength * relevance * rule.confidence

        # 2. Spreading Associative Activation across Rule Edges
        final_scores: dict[str, float] = dict(base_scores)
        edge_rows = conn.execute("SELECT rule_id_a, rule_id_b, weight FROM rule_edges").fetchall()
        
        for er in edge_rows:
            id_a, id_b, weight = er["rule_id_a"], er["rule_id_b"], float(er["weight"])
            if id_a in base_scores and id_b in final_scores:
                # Activation spreads from id_a to id_b
                spread_from_a = base_scores[id_a] * weight * self.config.spreading_activation_factor
                final_scores[id_b] += spread_from_a

        # 3. Rank rules descending by combined score
        ranked = [
            (rules_by_id[rule_id], score)
            for rule_id, score in final_scores.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_rules = ranked[:limit]

        # 4. Apply Hebbian potentiation to recalled memories
        if reinforce and top_rules:
            with conn:
                for rule, _ in top_rules:
                    new_strength = min(1.0, rule.base_strength + self.config.reinforcement_boost)
                    rule.access_count += 1
                    rule.last_accessed_at = now
                    rule.base_strength = new_strength
                    conn.execute(
                        """
                        UPDATE cortical_rules
                        SET access_count = access_count + 1,
                            last_accessed_at = ?,
                            base_strength = ?
                        WHERE id = ?
                        """,
                        (now, new_strength, rule.id)
                    )

        return top_rules

    def prune_decayed_rules(self, current_time: float | None = None) -> int:
        """Prunes rules whose decayed effective strength has fallen below prune_threshold and logs audit."""
        import uuid
        now = current_time or datetime.now(timezone.utc).timestamp()
        conn = self.db.get_connection()
        rows = conn.execute("SELECT * FROM cortical_rules").fetchall()

        pruned_count = 0
        with conn:
            for r in rows:
                rule = self._row_to_rule(r)
                strength = rule.calculate_effective_strength(
                    decay_rate=self.config.default_decay_rate,
                    current_time=now
                )
                if strength < self.config.prune_threshold:
                    audit_id = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO forgotten_audit (
                            id, item_type, original_id, signature_or_text, reason, category, metadata_json, forgotten_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (audit_id, "rule", rule.id, rule.rule_text, f"decay_pruned (strength={strength:.3f})", rule.category, None, now)
                    )
                    conn.execute("DELETE FROM cortical_rules WHERE id = ?", (rule.id,))
                    conn.execute("DELETE FROM rule_edges WHERE rule_id_a = ? OR rule_id_b = ?", (rule.id, rule.id))
                    if rule.id in self._vec_cache:
                        del self._vec_cache[rule.id]
                    pruned_count += 1

        return pruned_count

    def forget_rule(self, rule_id: str, reason: str = "manual") -> bool:
        """Explicitly forgets / deletes a cortical rule by ID and logs the audit trail."""
        import uuid
        now = datetime.now(timezone.utc).timestamp()
        conn = self.db.get_connection()
        with conn:
            r = conn.execute("SELECT * FROM cortical_rules WHERE id = ?", (rule_id,)).fetchone()
            if r:
                audit_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO forgotten_audit (
                        id, item_type, original_id, signature_or_text, reason, category, metadata_json, forgotten_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (audit_id, "rule", rule_id, r["rule_text"], reason, r["category"], None, now)
                )
            cur = conn.execute("DELETE FROM cortical_rules WHERE id = ?", (rule_id,))
            conn.execute("DELETE FROM rule_edges WHERE rule_id_a = ? OR rule_id_b = ?", (rule_id, rule_id))
            if rule_id in self._vec_cache:
                del self._vec_cache[rule_id]
            return cur.rowcount > 0

    def get_forgotten_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieves audit trail of forgotten or pruned rules and anti-memories."""
        conn = self.db.get_connection()
        rows = conn.execute(
            "SELECT * FROM forgotten_audit ORDER BY forgotten_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_edges(self, rule_id: str | None = None) -> list[RuleEdge]:
        """Returns associative edges for a given rule or for the whole graph."""
        conn = self.db.get_connection()
        if rule_id:
            rows = conn.execute(
                "SELECT * FROM rule_edges WHERE rule_id_a = ? OR rule_id_b = ?",
                (rule_id, rule_id)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM rule_edges").fetchall()
        
        return [
            RuleEdge(
                rule_id_a=r["rule_id_a"],
                rule_id_b=r["rule_id_b"],
                weight=float(r["weight"]),
                created_at=float(r["created_at"])
            )
            for r in rows
        ]

    def count(self) -> int:
        conn = self.db.get_connection()
        return conn.execute("SELECT COUNT(*) FROM cortical_rules").fetchone()[0]

    def _create_edge(self, id_a: str, id_b: str, weight: float, created_at: float):
        """Creates bidirectional edge in rule_edges graph."""
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO rule_edges (rule_id_a, rule_id_b, weight, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(rule_id_a, rule_id_b) DO UPDATE SET weight = excluded.weight
                """,
                (id_a, id_b, weight, created_at)
            )
            conn.execute(
                """
                INSERT INTO rule_edges (rule_id_a, rule_id_b, weight, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(rule_id_a, rule_id_b) DO UPDATE SET weight = excluded.weight
                """,
                (id_b, id_a, weight, created_at)
            )

    @staticmethod
    def _row_to_rule(r) -> CorticalRule:
        embedding = json.loads(r["embedding"]) if ("embedding" in r and r["embedding"]) else None
        source_ids = json.loads(r["source_episode_ids"]) if ("source_episode_ids" in r and r["source_episode_ids"]) else []
        valence = ValenceType(r["valence"]) if ("valence" in r and r["valence"]) else ValenceType.POSITIVE

        return CorticalRule(
            id=r["id"],
            rule_text=r["rule_text"],
            category=r["category"],
            confidence=r["confidence"],
            access_count=r["access_count"],
            created_at=r["created_at"],
            last_accessed_at=r["last_accessed_at"],
            base_strength=r["base_strength"],
            valence=valence,
            embedding=embedding,
            source_episode_ids=source_ids
        )
