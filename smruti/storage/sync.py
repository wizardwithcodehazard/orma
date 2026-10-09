"""
sync.py - Portable Cognitive Memory Bundling and Team Synchronization.
Allows exporting and importing verified cortical rules, anti-memories, and associative
graph edges across team members, CI environments, and central repositories.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smruti.storage.db import DatabaseManager
from smruti.storage.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class TeamMemoryBundle:
    """Manages serialization, export, and import of shared memory state."""

    @staticmethod
    def export_bundle(db: DatabaseManager, project_root: str | None = None) -> dict[str, Any]:
        """
        Exports active cortical rules, anti-memories, and associative graph edges into a portable dictionary.
        """
        conn = db.get_connection()
        now = datetime.now(timezone.utc).timestamp()

        # 1. Fetch cortical rules
        rules_rows = conn.execute("SELECT * FROM cortical_rules").fetchall()
        rules = [dict(r) for r in rules_rows]
        for r_dict in rules:
            raw_emb = r_dict.get("embedding")
            if raw_emb is not None:
                r_dict["embedding"] = EmbeddingEngine.blob_to_vec(raw_emb)

        # 2. Fetch active anti-memories
        if project_root:
            anti_rows = conn.execute(
                "SELECT * FROM anti_memories WHERE is_active = 1 AND (project_root IS NULL OR project_root = ?)",
                (project_root.strip(),)
            ).fetchall()
        else:
            anti_rows = conn.execute("SELECT * FROM anti_memories WHERE is_active = 1").fetchall()
        anti_memories = [dict(r) for r in anti_rows]
        for a_dict in anti_memories:
            raw_emb = a_dict.get("embedding")
            if raw_emb is not None:
                a_dict["embedding"] = EmbeddingEngine.blob_to_vec(raw_emb)

        # 3. Fetch associative graph edges
        edge_rows = conn.execute("SELECT * FROM rule_edges").fetchall()
        edges = [dict(r) for r in edge_rows]

        bundle = {
            "version": 2,
            "exported_at": now,
            "project_root": project_root,
            "summary": {
                "cortical_rules_count": len(rules),
                "anti_memories_count": len(anti_memories),
                "rule_edges_count": len(edges),
            },
            "cortical_rules": rules,
            "anti_memories": anti_memories,
            "rule_edges": edges,
        }
        logger.info(
            "Exported team memory bundle: %d rules, %d anti-memories, %d edges",
            len(rules), len(anti_memories), len(edges)
        )
        return bundle

    @staticmethod
    def import_bundle(
        db: DatabaseManager,
        bundle: dict[str, Any],
        overwrite: bool = False
    ) -> dict[str, int]:
        """
        Imports memory items from a bundle into the local database.
        Applies idempotency: updates existing signatures/IDs or skips duplicates.
        """
        conn = db.get_connection()
        imported_rules = 0
        imported_anti = 0
        imported_edges = 0

        # Import cortical rules
        rules = bundle.get("cortical_rules", [])
        with db.write_transaction(conn):
            for r in rules:
                emb_val = r.get("embedding")
                emb_blob = EmbeddingEngine.vec_to_blob(EmbeddingEngine.blob_to_vec(emb_val)) if emb_val is not None else None
                existing = conn.execute("SELECT id FROM cortical_rules WHERE id = ?", (r["id"],)).fetchone()
                if existing:
                    if overwrite:
                        conn.execute(
                            """
                            UPDATE cortical_rules
                            SET rule_text = ?, category = ?, confidence = ?, base_strength = ?,
                                valence = ?, embedding = ?, source_episode_ids = ?
                            WHERE id = ?
                            """,
                            (
                                r["rule_text"], r["category"], r["confidence"], r["base_strength"],
                                r.get("valence", "positive"), emb_blob,
                                r.get("source_episode_ids"), r["id"]
                            )
                        )
                        if db.vec_available and emb_blob:
                            conn.execute("DELETE FROM vec_cortical_rules WHERE rule_id = ?", (r["id"],))
                            conn.execute("INSERT INTO vec_cortical_rules (rule_id, embedding) VALUES (?, ?)", (r["id"], emb_blob))
                        imported_rules += 1
                else:
                    conn.execute(
                        """
                        INSERT INTO cortical_rules (
                            id, rule_text, category, confidence, access_count,
                            created_at, last_accessed_at, base_strength, valence,
                            embedding, source_episode_ids
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            r["id"], r["rule_text"], r["category"], r["confidence"],
                            r.get("access_count", 0), r.get("created_at", 0.0),
                            r.get("last_accessed_at", 0.0), r.get("base_strength", 1.0),
                            r.get("valence", "positive"), emb_blob,
                            r.get("source_episode_ids")
                        )
                    )
                    if db.vec_available and emb_blob:
                        conn.execute("DELETE FROM vec_cortical_rules WHERE rule_id = ?", (r["id"],))
                        conn.execute("INSERT INTO vec_cortical_rules (rule_id, embedding) VALUES (?, ?)", (r["id"], emb_blob))
                    imported_rules += 1

            # Import anti-memories
            anti_memories = bundle.get("anti_memories", [])
            for a in anti_memories:
                a_emb_val = a.get("embedding")
                a_emb_blob = EmbeddingEngine.vec_to_blob(EmbeddingEngine.blob_to_vec(a_emb_val)) if a_emb_val is not None else None
                conn.execute(
                    """
                    INSERT INTO anti_memories (
                        id, signature, pattern, reason, suggested_fix, context_tags,
                        times_triggered, severity, created_at, last_seen,
                        valence, project_root, is_active, embedding, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(signature) DO UPDATE SET
                        pattern = excluded.pattern,
                        reason = excluded.reason,
                        suggested_fix = excluded.suggested_fix,
                        severity = excluded.severity,
                        last_seen = excluded.last_seen,
                        project_root = excluded.project_root,
                        is_active = 1,
                        embedding = excluded.embedding,
                        expires_at = excluded.expires_at
                    """,
                    (
                        a["id"], a["signature"], a["pattern"], a["reason"],
                        a.get("suggested_fix"), a.get("context_tags"),
                        a.get("times_triggered", 0), a.get("severity", "high"),
                        a.get("created_at", 0.0), a.get("last_seen", 0.0),
                        a.get("valence", "inhibitory"), a.get("project_root"),
                        1 if a.get("is_active", True) else 0,
                        a_emb_blob, a.get("expires_at")
                    )
                )
                if db.vec_available and a_emb_blob:
                    conn.execute("DELETE FROM vec_anti_memories WHERE anti_memory_id = ?", (a["id"],))
                    conn.execute("INSERT INTO vec_anti_memories (anti_memory_id, embedding) VALUES (?, ?)", (a["id"], a_emb_blob))
                imported_anti += 1

            # Import rule edges
            edges = bundle.get("rule_edges", [])
            for e in edges:
                conn.execute(
                    """
                    INSERT INTO rule_edges (rule_id_a, rule_id_b, weight, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(rule_id_a, rule_id_b) DO UPDATE SET weight = excluded.weight
                    """,
                    (e["rule_id_a"], e["rule_id_b"], e["weight"], e.get("created_at", 0.0))
                )
                imported_edges += 1

        logger.info(
            "Imported team memory bundle: %d rules, %d anti-memories, %d edges",
            imported_rules, imported_anti, imported_edges
        )
        return {
            "imported_rules": imported_rules,
            "imported_anti_memories": imported_anti,
            "imported_edges": imported_edges,
        }

    @classmethod
    def export_file(
        cls,
        db: DatabaseManager,
        file_path: Path | str,
        project_root: str | None = None
    ) -> Path:
        """Serializes and saves a memory bundle to a JSON file."""
        target_path = Path(file_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = cls.export_bundle(db, project_root=project_root)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        return target_path

    @classmethod
    def import_file(
        cls,
        db: DatabaseManager,
        file_path: Path | str,
        overwrite: bool = False
    ) -> dict[str, int]:
        """Loads and imports a memory bundle from a JSON file."""
        source_path = Path(file_path).resolve()
        with open(source_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        return cls.import_bundle(db, bundle, overwrite=overwrite)
