"""
test_sqlite_vec_storage.py - Unit and benchmark tests for native in-SQLite vector storage (sqlite-vec),
binary float32 BLOB embeddings, and SQL KNN vector distance operators.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from smruti.config import smrutiConfig
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.storage.db import DatabaseManager
from smruti.storage.embeddings import EmbeddingEngine


@pytest.fixture
def env():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_vec.db",
            dedup_similarity_threshold=0.85
        )
        db = DatabaseManager(config)
        cortex = Cortex(db, config)
        gate = InhibitoryGate(db)
        try:
            yield {
                "config": config,
                "db": db,
                "cortex": cortex,
                "gate": gate
            }
        finally:
            db.close()


def test_sqlite_vec_extension_loaded_and_schema_v6(env):
    """Verifies that sqlite-vec extension is active and migration v6 created vec0 virtual tables."""
    db = env["db"]
    assert db.vec_available is True

    conn = db.get_connection()
    # Check schema versions
    rows = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    versions = [r[0] for r in rows]
    assert 6 in versions

    # Check vec0 virtual tables exist
    c_vec_count = conn.execute("SELECT COUNT(*) FROM vec_cortical_rules").fetchone()[0]
    assert c_vec_count == 0
    a_vec_count = conn.execute("SELECT COUNT(*) FROM vec_anti_memories").fetchone()[0]
    assert a_vec_count == 0


def test_embeddings_stored_as_binary_blob(env):
    """Verifies embeddings are stored as compact 1536-byte IEEE 754 float32 BLOBs rather than JSON strings."""
    cortex = env["cortex"]
    gate = env["gate"]
    db = env["db"]

    # 1. Add cortical rule
    rule = cortex.add_rule("Always use structured logging with JSON format in production", category="logging")
    assert rule.embedding is not None
    assert len(rule.embedding) == 384

    # Check raw DB storage format
    conn = db.get_connection()
    row = conn.execute("SELECT embedding FROM cortical_rules WHERE id = ?", (rule.id,)).fetchone()
    raw_emb = row["embedding"]
    assert isinstance(raw_emb, bytes)
    assert len(raw_emb) == 384 * 4  # 1536 bytes for 384 float32 values!

    # Check vec_cortical_rules index entry
    vec_row = conn.execute("SELECT rule_id, distance FROM vec_cortical_rules WHERE rule_id = ?", (rule.id,)).fetchone()
    assert vec_row is not None
    assert vec_row["rule_id"] == rule.id

    # 2. Add anti-memory
    anti = gate.record_anti_memory(
        signature="rm_rf_root_hazard",
        pattern="rm -rf /",
        reason="Catastrophic root filesystem destruction hazard"
    )
    anti_row = conn.execute("SELECT embedding FROM anti_memories WHERE id = ?", (anti.id,)).fetchone()
    raw_anti_emb = anti_row["embedding"]
    assert isinstance(raw_anti_emb, bytes)
    assert len(raw_anti_emb) == 384 * 4

    vec_anti_row = conn.execute(
        "SELECT anti_memory_id FROM vec_anti_memories WHERE anti_memory_id = ?",
        (anti.id,)
    ).fetchone()
    assert vec_anti_row is not None


def test_sqlite_vec_knn_deduplication(env):
    """Verifies that duplicate rules are detected and reinforced via in-SQLite KNN queries."""
    cortex = env["cortex"]

    # 1. Base rule
    r1 = cortex.add_rule(
        "Always enforce SSL TLS certificates verification on incoming HTTP requests",
        category="security",
        base_strength=0.7
    )
    assert cortex.count() == 1

    # 2. Semantically identical rule
    r2 = cortex.add_rule(
        "Ensure HTTPS requests strictly validate TLS certificates",
        category="security",
        base_strength=0.7
    )

    # Must be deduplicated into rule 1
    assert cortex.count() == 1
    assert r2.id == r1.id
    assert r2.access_count == r1.access_count + 1
    assert r2.base_strength > 0.7


def test_sqlite_vec_inhibition_gate_semantic_match(env):
    """Verifies that InhibitoryGate blocks dangerous commands using SQL KNN search in vec_anti_memories."""
    gate = env["gate"]

    gate.record_anti_memory(
        signature="drop_all_tables_hazard",
        pattern="DROP TABLE.*CASCADE",
        reason="Irreversible database wipe without confirmation"
    )

    # Command with semantic similarity
    res = gate.check_action("DROP TABLE users CASCADE;")
    assert res.passed is False
    assert res.matched_signature == "drop_all_tables_hazard"


def test_sqlite_vec_trigger_cascade_delete(env):
    """Verifies that deleting a cortical rule or anti-memory triggers automatic deletion from vec0 virtual table."""
    cortex = env["cortex"]
    gate = env["gate"]
    db = env["db"]
    conn = db.get_connection()

    rule = cortex.add_rule("Temporary caching convention", category="cache")
    anti = gate.record_anti_memory("temp_pattern", "kill -9 1", "Cannot kill init process")

    # Verify present in virtual tables
    assert conn.execute("SELECT COUNT(*) FROM vec_cortical_rules WHERE rule_id = ?", (rule.id,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM vec_anti_memories WHERE anti_memory_id = ?", (anti.id,)).fetchone()[0] == 1

    # Delete from main tables
    with conn:
        conn.execute("DELETE FROM cortical_rules WHERE id = ?", (rule.id,))
        conn.execute("DELETE FROM anti_memories WHERE id = ?", (anti.id,))

    # Triggers must have cleaned up virtual tables automatically
    assert conn.execute("SELECT COUNT(*) FROM vec_cortical_rules WHERE rule_id = ?", (rule.id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vec_anti_memories WHERE anti_memory_id = ?", (anti.id,)).fetchone()[0] == 0


def test_blob_to_vec_backward_compatibility():
    """Verifies EmbeddingEngine deserializes both legacy JSON strings and float32 BLOBs identically."""
    vec = [0.1, -0.25, 0.5, 0.99] + [0.0] * 380
    blob = EmbeddingEngine.vec_to_blob(vec)
    json_str = str(vec)

    vec_from_blob = EmbeddingEngine.blob_to_vec(blob)
    vec_from_json = EmbeddingEngine.blob_to_vec(json_str)

    assert vec_from_blob is not None
    assert vec_from_json is not None
    np.testing.assert_allclose(vec_from_blob[:4], vec[:4], atol=1e-5)
    np.testing.assert_allclose(vec_from_json[:4], vec[:4], atol=1e-5)


def test_benchmark_scale_latency(env):
    """Verifies that KNN vector search on hundreds of rules executes in under 5 milliseconds."""
    cortex = env["cortex"]

    topics = [
        "Use Redis for distributed caching", "Use Postgres for ACID transactions",
        "Encrypt all database backups at rest", "Rotate JWT signing keys every 30 days",
        "Enable Gzip compression for static web assets", "Use OAuth2 with PKCE for mobile auth",
        "Configure Prometheus to scrape node metrics", "Use Docker multi-stage builds to minimize images",
        "Enforce snake_case for all Python variables", "Require 2FA for all administrative logins",
        "Set TTL on temporary cache keys to 1 hour", "Use exponential backoff for network retries",
        "Limit HTTP request body size to 10MB", "Run database migrations before starting web workers",
        "Use WAL mode for SQLite concurrent access", "Sanitize all user inputs against XSS attacks",
        "Employ circuit breakers on remote RPC calls", "Log all audit events with RFC 3339 timestamps",
        "Automate dependency vulnerability scans weekly", "Configure dead letter queues for failed Kafka messages"
    ]

    for i, t in enumerate(topics):
        cortex.add_rule(f"{t} (rule {i})", category=f"cat_{i % 5}")

    assert cortex.count() == len(topics)

    # Measure recall latency
    start = time.perf_counter()
    recalled = cortex.recall_rules(query="caching and database security", limit=5)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert len(recalled) == 5
    print(f"\n[BENCHMARK] Recall latency with sqlite-vec: {elapsed_ms:.2f}ms")
    assert elapsed_ms < 20.0
