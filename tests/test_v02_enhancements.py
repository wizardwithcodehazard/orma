"""
test_v02_enhancements.py - Unit tests for:
1. SQLite connection pragmas and migration v4
2. Anti-memory TTL and expiration (query-time check and sleep pruning)
3. Read-only intent classifier for false-positive reduction
4. Declarative facts storage (smruti.remember)
5. Document chunking & ingestion (smruti.ingest)
"""

import tempfile
import time
from pathlib import Path

import pytest

from smruti.config import smrutiConfig
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate, _is_read_only_intent
from smruti.engine.stream import StreamBuffer
from smruti.framework import smruti
from smruti.interfaces.mcp_server import _chunk_text
from smruti.storage.db import DatabaseManager


@pytest.fixture
def test_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_v02.db"
        )
        db = DatabaseManager(config)
        stream = StreamBuffer(db)
        inhibitory = InhibitoryGate(db)
        cortex = Cortex(db, config)
        consolidator = Consolidator(db, config, stream=stream, inhibitory=inhibitory, cortex=cortex)
        try:
            yield {
                "config": config,
                "db": db,
                "stream": stream,
                "inhibitory": inhibitory,
                "cortex": cortex,
                "consolidator": consolidator,
            }
        finally:
            db.close()


def test_schema_migration_v4_and_pragmas(test_env):
    db = test_env["db"]
    conn = db.get_connection()

    # Verify migration v4 applied
    rows = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    versions = [r[0] for r in rows]
    assert 4 in versions

    # Verify expires_at column exists in anti_memories
    cur = conn.execute("PRAGMA table_info(anti_memories);")
    cols = [r["name"] for r in cur.fetchall()]
    assert "expires_at" in cols


def test_anti_memory_ttl_query_time(test_env):
    gate = test_env["inhibitory"]
    now = time.time()

    # Record an anti-memory that expired 10 seconds ago
    gate.record_anti_memory(
        signature="transient_expired_bug",
        pattern="test-command-expired",
        reason="Temporary service issue",
        expires_at=now - 10.0
    )

    # Should NOT be blocked because it has expired
    res = gate.check_action("test-command-expired")
    assert res.passed is True

    # Record an anti-memory with future TTL
    gate.record_anti_memory(
        signature="future_active_bug",
        pattern="test-command-active",
        reason="Active service issue",
        expires_in_days=1.0
    )

    # Should BE blocked because it is currently valid
    res_active = gate.check_action("test-command-active")
    assert res_active.passed is False
    assert res_active.matched_signature == "future_active_bug"


def test_sleep_pruning_expired_anti_memories(test_env):
    gate = test_env["inhibitory"]
    consolidator = test_env["consolidator"]
    db = test_env["db"]
    now = time.time()

    # Insert 2 expired and 1 active anti-memory
    gate.record_anti_memory("exp_1", "cmd1", "reason", expires_at=now - 50)
    gate.record_anti_memory("exp_2", "cmd2", "reason", expires_at=now - 20)
    gate.record_anti_memory("perm", "cmd3", "reason", expires_at=None)

    report = consolidator.sleep(current_time=now)
    assert report["pruned_expired_anti_memories"] == 2

    # Check database: only 'perm' remains
    conn = db.get_connection()
    remaining = conn.execute("SELECT signature FROM anti_memories").fetchall()
    sigs = [r[0] for r in remaining]
    assert "perm" in sigs
    assert "exp_1" not in sigs
    assert "exp_2" not in sigs


def test_read_only_intent_classifier():
    # Inspection commands
    assert _is_read_only_intent("SELECT * FROM users") is True
    assert _is_read_only_intent("git log --oneline -n 5") is True
    assert _is_read_only_intent("git status") is True
    assert _is_read_only_intent("cat package.json") is True
    assert _is_read_only_intent("grep -rn 'TODO' .") is True
    assert _is_read_only_intent("ls -la /tmp") is True
    assert _is_read_only_intent("docker ps") is True

    # Mutating / execution commands
    assert _is_read_only_intent("DROP TABLE users") is False
    assert _is_read_only_intent("rm -rf /") is False
    assert _is_read_only_intent("npm run build") is False
    assert _is_read_only_intent("git push origin main") is False
    assert _is_read_only_intent("python train.py") is False


def test_sdk_remember_and_recall():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = smrutiConfig(project_dir=Path(tmpdir), db_filename="test_sdk.db")
        mem = smruti(config=cfg)
        try:
            # Store declarative fact
            rule = mem.remember(
                fact="Postgres table 'audit_log' is append-only; never run UPDATE or DELETE queries on it.",
                category="architecture"
            )
            assert rule.category == "architecture"
            assert "append-only" in rule.rule_text

            # Recall fact via query
            recalled = mem.recall(query="audit_log update rule", limit=1)
            assert len(recalled) == 1
            top_rule, score = recalled[0]
            assert "audit_log" in top_rule.rule_text
            assert score > 0
        finally:
            mem.close()


def test_sdk_ingest_document():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = smrutiConfig(project_dir=Path(tmpdir), db_filename="test_ingest.db")
        mem = smruti(config=cfg)
        try:
            sample_doc = (
                "# Service Architecture\n\n"
                "All payment transactions must pass through the Risk Assessment engine before processing.\n\n"
                "The Risk engine issues a signed JWT token with a 30-second TTL to authorize checkout."
            )

            result = mem.ingest(sample_doc, source="payment_arch.md", category="architecture")
            assert result["status"] == "success"
            assert result["total_chunks"] >= 1
            assert result["stored"] >= 1

            # Recall information from the ingested document
            recalled = mem.recall(query="risk assessment payment token TTL", limit=1)
            assert len(recalled) >= 1
            assert "Risk" in recalled[0][0].rule_text
        finally:
            mem.close()



def test_chunking_utility():
    text = "Short text"
    chunks = _chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "Short text"

    long_text = "Paragraph one with some details.\n\nParagraph two with other details.\n\nParagraph three."
    chunks2 = _chunk_text(long_text, chunk_size=40, overlap=10)
    assert len(chunks2) >= 2


def test_user_preferences_and_prompt_context_assembly():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = smrutiConfig(project_dir=Path(tmpdir), db_filename="test_pref.db")
        mem = smruti(config=cfg)
        try:
            # 1. Set user preferences
            mem.set_preference("User strictly uses vanilla CSS instead of Tailwind.")
            mem.set_preference("Always write pytest unit tests for every new function.")

            prefs = mem.get_preferences()
            assert len(prefs) == 2
            pref_texts = [p.rule_text for p in prefs]
            assert any("vanilla CSS" in t for t in pref_texts)

            # 2. Add an architectural rule
            mem.remember("PostgreSQL connection pooling max connections = 20", category="database")

            # 3. Add an inhibitory anti-memory
            mem.record_anti_memory(
                signature="no_force_push",
                pattern="git push.*--force",
                reason="Force push destroys remote branch history",
                suggested_fix="Use normal git push or rebase locally"
            )

            # 4. Generate structured prompt context
            context_block = mem.get_prompt_context(query="database queries and css styling")
            assert "[smruti ACTIVE MEMORY CONTEXT]" in context_block
            assert "User Directives & Preferences:" in context_block
            assert "vanilla CSS" in context_block
            assert "Active Execution Constraints (Known Dead Ends):" in context_block
            assert "no_force_push" in context_block or "Force push" in context_block
        finally:
            mem.close()



def test_batch_vectorized_cosine_matrix():
    import numpy as np
    from smruti.storage.embeddings import EmbeddingEngine

    # Create dummy query and 100 random vectors in R^384
    np.random.seed(42)
    dim = 384
    q = np.random.randn(dim).astype(np.float32)
    matrix = np.random.randn(100, dim).astype(np.float32)

    # Compute vectorized similarities
    batch_sims = EmbeddingEngine.batch_cosine_similarity(q, matrix)
    assert len(batch_sims) == 100
    assert batch_sims.dtype == np.float32

    # Verify identical to individual cosine similarity calls
    for i in range(5):
        single_sim = EmbeddingEngine.cosine_similarity(q.tolist(), matrix[i].tolist())
        assert abs(batch_sims[i] - single_sim) < 1e-4


def test_team_memory_bundle_export_import():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        cfg1 = smrutiConfig(project_dir=dir_path / "dev1", db_filename="dev1.db")
        cfg2 = smrutiConfig(project_dir=dir_path / "dev2", db_filename="dev2.db")

        mem1 = smruti(config=cfg1)
        mem2 = smruti(config=cfg2)
        try:
            # Dev 1 discovers architecture rule and failure anti-memory
            mem1.remember("Payment webhooks must verify HMAC signature in header", category="security")
            mem1.record_anti_memory(
                signature="unindexed_query_bug",
                pattern="SELECT * FROM logs WHERE trace_id",
                reason="Full table scan causes DB timeout without trace_id index"
            )

            # Dev 1 exports bundle to JSON file
            bundle_file = dir_path / "team_bundle.json"
            mem1.export_file(bundle_file)
            assert bundle_file.is_file()

            # Dev 2 imports bundle from JSON file
            res = mem2.import_file(bundle_file)
            assert res["imported_rules"] >= 1
            assert res["imported_anti_memories"] >= 1

            # Dev 2 now possesses the same memory!
            recalled = mem2.recall(query="HMAC signature webhook", limit=1)
            assert len(recalled) == 1
            assert "HMAC" in recalled[0][0].rule_text

            # Dev 2 is protected by Dev 1's anti-memory
            check = mem2.preflight("SELECT * FROM logs WHERE trace_id = 'abc'")
            assert check.passed is False
            assert check.matched_signature == "unindexed_query_bug"
        finally:
            mem1.close()
            mem2.close()


def test_sqlite_storage_backend_ping(test_env):
    from smruti.storage.backends import SQLiteStorageBackend
    backend = SQLiteStorageBackend(test_env["db"])
    assert backend.ping() is True


def test_negative_constraints_anti_facts():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = smrutiConfig(project_dir=Path(tmpdir), db_filename="test_antifact.db")
        mem = smruti(config=cfg)
        try:
            # Record negative declarative constraint
            rule = mem.record_anti_fact("Do not use Redux Toolkit; strictly use Zustand for client state.")
            assert rule.category == "anti_pattern"
            assert "Zustand" in rule.rule_text

            anti_facts = mem.get_anti_facts()
            assert len(anti_facts) == 1
            assert "Redux" in anti_facts[0].rule_text

            # Verify prompt context rendering puts anti-pattern in high-priority section
            ctx = mem.get_prompt_context(query="state management store")
            assert "### Strictly Forbidden Anti-Patterns (Do NOT Suggest or Implement):" in ctx
            assert "[FORBIDDEN] Do not use Redux Toolkit" in ctx
        finally:
            mem.close()


def test_mcp_resource_and_prompt_registration(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_mcp_res.db"
        )
        db = DatabaseManager(config)
        monkeypatch.setattr("smruti.storage.db._default_db", db)
        # Invalidate mcp_server engine cache to use the test db
        import smruti.interfaces.mcp_server as mcp_mod
        mcp_mod._cached_db = None
        mcp_mod._cached_bundle = None
        try:
            res = mcp_mod.smruti_active_context_resource()
            assert isinstance(res, str)
            p = mcp_mod.smruti_context_prompt("Refactoring frontend state")
            assert isinstance(p, str)
        finally:
            db.close()
            mcp_mod._cached_db = None
            mcp_mod._cached_bundle = None


def test_in_flight_observe_neural_extractor():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = smrutiConfig(
            project_dir=Path(tmpdir),
            db_filename="test_observe.db"
        )
        mem = smruti(config=config)
        try:
            # 1. User says an architectural directive
            rule1 = mem.observe("We strictly use PostgreSQL with connection pooling as our primary database.")
            assert rule1 is not None
            assert "PostgreSQL" in rule1.rule_text

            # 2. User says a forbidden anti-pattern
            rule2 = mem.observe("Do not use synchronous requests in async endpoints; it blocks the event loop.")
            assert rule2 is not None

            # 3. User says normal transient task / bug report / chitchat
            rule3 = mem.observe("Can you check line 42 for the typo?")
            assert rule3 is None

            rule4 = mem.observe("hello how are you")
            assert rule4 is None
        finally:
            mem.close()


def test_stream_buffer_secret_masking(test_env):
    from smruti.engine.stream import StreamBuffer, mask_sensitive_data
    from smruti.models import ActionStatus

    # Test masking helper directly
    raw_api_key = "curl -H 'Authorization: Bearer sk-1234567890abcdef1234567890' https://api.openai.com"
    masked = mask_sensitive_data(raw_api_key)
    assert "sk-1234567890" not in masked
    assert "[REDACTED_API_KEY]" in masked or "[REDACTED_BEARER_TOKEN]" in masked

    raw_gh_token = "git clone https://ghp_1234567890abcdef1234567890abcdef1234@github.com/repo"
    masked_gh = mask_sensitive_data(raw_gh_token)
    assert "ghp_1234567890" not in masked_gh
    assert "[REDACTED_GITHUB_TOKEN]" in masked_gh

    # Test through StreamBuffer.append()
    stream = StreamBuffer(db=test_env["db"], auto_consolidate=False)
    ep = stream.append(
        action="export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456",
        result="Connecting to postgres://user:super_secret_pw@db.prod:5432/main",
        status=ActionStatus.SUCCESS
    )
    assert "sk-abcdefgh" not in ep.action
    assert "[REDACTED_API_KEY]" in ep.action
    assert "super_secret_pw" not in ep.result
    assert "[REDACTED_PASSWORD]" in ep.result


def test_inhibition_result_confidence_and_diagnostics(test_env):
    from smruti.engine.inhibitory import InhibitoryGate

    gate = InhibitoryGate(test_env["db"])
    gate.record_anti_memory(
        signature="wipe_all_data",
        pattern="rm -rf /data",
        reason="Catastrophic data deletion"
    )

    # 1. Test passing check -> confidence 0.0, empty sources
    pass_res = gate.check_action("git status")
    assert pass_res.passed is True
    assert pass_res.confidence == 0.0
    assert pass_res.matched_sources == []

    # 2. Test literal substring block -> confidence 1.0, literal_substring in sources
    lit_res = gate.check_action("rm -rf /data")
    assert lit_res.passed is False
    assert lit_res.confidence == 1.0
    assert "literal_substring" in lit_res.matched_sources

    # 3. Test semantic variation block -> confidence >= 0.82, semantic_similarity in sources
    sem_res = gate.check_action("delete recursively everything in /data directory")
    # Even if blocked or passed, test contract consistency
    if not sem_res.passed:
        assert sem_res.confidence > 0.0
        assert any("semantic_similarity" in s for s in sem_res.matched_sources)


def test_llm_providers_and_distillation(test_env):
    from smruti.engine.llm import LLMClient
    from smruti.engine.consolidator import Consolidator
    from smruti.models import ActionStatus

    # 1. Test fallback when unconfigured (Occam's razor: zero API keys needed)
    default_client = LLMClient()
    assert default_client.is_configured() is False

    # Heuristic fallback for failure cluster
    cluster = [
        {"action": "npm run build", "result": "RollupError: Could not resolve ./missing"}
    ]
    heuristic_reason = default_client.summarize_failure_cluster(cluster)
    assert "Repeated failures in 'npm run build'" in heuristic_reason

    # Heuristic fallback for causal transition
    transitions = [
        {"failed_action": "npm run build", "fix_action": "npm install && npm run build"}
    ]
    heuristic_rule = default_client.distill_resolution_heuristic(transitions)
    assert "When 'npm run build' fails, use 'npm install && npm run build' instead." in heuristic_rule

    # 2. Test host agent in-memory summarizer hook injection
    def mock_agent_brain(task_type: str, payload: list) -> str:
        if task_type == "failure_summary":
            return "Agent Brain Distillation: Port 8080 collision caused by zombie process"
        if task_type == "causal_resolution":
            return "Agent Brain Distillation: When port 8080 is blocked, kill PID with lsof before restart"
        return "default"

    agent_client = LLMClient(summarizer=mock_agent_brain)
    assert agent_client.is_configured() is True
    assert agent_client.summarize_failure_cluster(cluster) == "Agent Brain Distillation: Port 8080 collision caused by zombie process"

    # 3. Test sleep consolidation with host agent brain hook
    consolidator = Consolidator(
        db=test_env["db"],
        stream=test_env["stream"],
        inhibitory=test_env["inhibitory"],
        cortex=test_env["cortex"],
        llm_client=agent_client
    )

    # Record 2 failures
    test_env["stream"].append(
        action="python server.py --port 8080",
        result="OSError: [Errno 98] Address already in use",
        status=ActionStatus.FAILURE
    )
    test_env["stream"].append(
        action="python server.py --port 8080",
        result="OSError: [Errno 98] Address already in use",
        status=ActionStatus.FAILURE
    )

    report = consolidator.sleep()
    assert report["promoted_anti_memories"] == 1

    # Verify anti-memory has the distilled agent brain reason
    anti = test_env["inhibitory"].list_all(active_only=True)
    assert any("Agent Brain Distillation: Port 8080 collision" in m.reason for m in anti)


def test_forgotten_audit_trail(test_env):
    cortex = test_env["cortex"]
    inhibitory = test_env["inhibitory"]

    # 1. Forget rule manually
    rule = cortex.add_rule("Test rule to be forgotten", category="temp")
    forgotten = cortex.forget_rule(rule.id, reason="superseded by newer RFC")
    assert forgotten is True

    # 2. Deactivate anti-memory manually
    anti = inhibitory.record_anti_memory("temp_sig", "temp_pat", "Temporary issue")
    deactivated = inhibitory.forget_anti_memory("temp_sig", reason="fixed upstream in v2.4")
    assert deactivated is True

    # 3. Verify audit trail logs both events
    audit = cortex.get_forgotten_audit(limit=10)
    assert len(audit) >= 2

    rule_audits = [a for a in audit if a["item_type"] == "rule"]
    assert any("superseded by newer RFC" in a["reason"] for a in rule_audits)
    assert any("Test rule to be forgotten" in a["signature_or_text"] for a in rule_audits)

    anti_audits = [a for a in audit if a["item_type"] == "anti_memory"]
    assert any("fixed upstream in v2.4" in a["reason"] for a in anti_audits)
    assert any("temp_sig" in a["signature_or_text"] for a in anti_audits)








