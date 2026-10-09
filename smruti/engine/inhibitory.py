"""
inhibitory.py - Tier 3B: Inhibitory Anti-Memories (Vimarsha).
Preflight gate that intercepts agent actions before execution if a matching failure signature is found.
Features:
1. Fast-path literal containment
2. Dense vector neural semantic matching (replaces brittle regex)
3. Optional LLM arbiter evaluation for nuanced semantic decisions
4. Project scoping and dynamic memory retraction (forget)
"""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone

from smruti.models import AntiMemory, InhibitionResult, ValenceType
from smruti.storage.db import DatabaseManager, get_db
from smruti.storage.embeddings import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)

REGEX_CHARS = set(r"^$.*+?{}[]\|()")

# First tokens that strongly signal a non-mutating, read-only operation.
# These bypass destructive anti-memories unless similarity is extremely high (>=0.95).
_READ_ONLY_VERBS: frozenset[str] = frozenset({
    # SQL read
    "select", "with",
    # HTTP read
    "get",
    # Shell inspection
    "ls", "ll", "dir", "cat", "bat", "less", "more", "head", "tail",
    "grep", "rg", "find", "locate", "which", "whereis", "type",
    "stat", "file", "wc", "diff", "md5sum", "sha256sum",
    "describe", "explain", "show", "inspect", "info", "status",
    "ps", "top", "htop", "df", "du", "lsof", "netstat", "ss",
    "ping", "traceroute", "nslookup", "dig",
    "echo", "print", "printf", "pwd", "env", "printenv",
    "git log", "git status", "git diff", "git show", "git branch",
    # Python/Node inspection
    "python -c", "node -e",
    # Docker read
    "docker ps", "docker images", "docker logs", "docker inspect",
})

_READ_ONLY_THRESHOLD = 0.95  # Much stricter semantic threshold for read-only actions


def _is_read_only_intent(action: str) -> bool:
    """Returns True if the action's first verb strongly signals a non-mutating read operation."""
    lower = action.strip().lower()
    # Check two-token prefixes first (e.g. "git log", "docker ps")
    for verb in _READ_ONLY_VERBS:
        if " " in verb and lower.startswith(verb):
            return True
    # Single-token first word
    first_word = lower.split()[0] if lower.split() else ""
    return first_word in _READ_ONLY_VERBS


class InhibitoryGate:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        llm_evaluator: Callable[[str, AntiMemory], tuple[bool, str | None]] | None = None,
        semantic_threshold: float = 0.82
    ):
        self.db = db or get_db()
        self.embeddings = embedding_engine or get_embedding_engine()
        self.llm_evaluator = llm_evaluator
        self.semantic_threshold = semantic_threshold
        self._compiled_patterns: dict[str, re.Pattern | None] = {}

    def record_anti_memory(
        self,
        signature: str,
        pattern: str,
        reason: str,
        suggested_fix: str | None = None,
        context_tags: list[str] | None = None,
        severity: str = "high",
        project_root: str | None = None,
        expires_at: float | None = None,
        expires_in_days: float | None = None,
    ) -> AntiMemory:
        """Records or updates a known failure signature with dense vector embeddings, project scoping, and TTL."""
        clean_sig = signature.strip()
        clean_pat = pattern.strip()
        clean_reason = reason.strip()

        if expires_in_days is not None and expires_in_days > 0 and expires_at is None:
            expires_at = datetime.now(timezone.utc).timestamp() + expires_in_days * 86400.0

        # Compute neural embedding over signature, pattern, and reason
        embed_payload = f"{clean_sig} {clean_pat} {clean_reason}"
        vec = self.embeddings.embed_text(embed_payload)
        vec_blob = EmbeddingEngine.vec_to_blob(vec)

        anti_memory = AntiMemory(
            signature=clean_sig,
            pattern=clean_pat,
            reason=clean_reason,
            suggested_fix=suggested_fix.strip() if suggested_fix else None,
            context_tags=context_tags or [],
            severity=severity,
            valence=ValenceType.INHIBITORY,
            project_root=project_root.strip() if project_root else None,
            is_active=True,
            embedding=vec,
            expires_at=expires_at
        )

        conn = self.db.get_connection()
        with conn:
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
                    context_tags = excluded.context_tags,
                    severity = excluded.severity,
                    last_seen = excluded.last_seen,
                    project_root = excluded.project_root,
                    is_active = 1,
                    embedding = excluded.embedding,
                    expires_at = excluded.expires_at
                """,
                (
                    anti_memory.id,
                    anti_memory.signature,
                    anti_memory.pattern,
                    anti_memory.reason,
                    anti_memory.suggested_fix,
                    json.dumps(anti_memory.context_tags),
                    anti_memory.times_triggered,
                    anti_memory.severity,
                    anti_memory.created_at,
                    anti_memory.last_seen,
                    anti_memory.valence.value,
                    anti_memory.project_root,
                    1 if anti_memory.is_active else 0,
                    vec_blob,
                    anti_memory.expires_at
                )
            )

            # Insert or update into vec_anti_memories virtual table
            if self.db.vec_available and vec_blob:
                conn.execute("DELETE FROM vec_anti_memories WHERE anti_memory_id = ?", (anti_memory.id,))
                conn.execute(
                    "INSERT INTO vec_anti_memories (anti_memory_id, embedding) VALUES (?, ?)",
                    (anti_memory.id, vec_blob)
                )

        logger.info("Recorded anti-memory '%s' (project: %s, expires_at: %s)", clean_sig, project_root, expires_at)
        return anti_memory

    def check_action(
        self,
        action: str,
        context: str = "",
        project_root: str | None = None
    ) -> InhibitionResult:
        """
        Active Preflight Gate: Evaluates candidate action against active anti-memories.
        1. Fast substring containment.
        2. Neural vector semantic similarity match.
        3. Optional LLM arbiter for nuanced semantic judgment.
        """
        conn = self.db.get_connection()
        normalized_action = action.strip()
        combined_text = f"{context} {normalized_action}".lower()

        now = datetime.now(timezone.utc).timestamp()
        # SQL level filter: Only active, non-expired anti-memories for this project (or global)
        if project_root:
            query = """
                SELECT * FROM anti_memories 
                WHERE is_active = 1 
                  AND (project_root IS NULL OR project_root = ?)
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY times_triggered DESC
            """
            rows = conn.execute(query, (project_root.strip(), now)).fetchall()
        else:
            query = """
                SELECT * FROM anti_memories 
                WHERE is_active = 1
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY times_triggered DESC
            """
            rows = conn.execute(query, (now,)).fetchall()

        if not rows:
            return InhibitionResult(passed=True)

        # Compute candidate action embedding for neural semantic matching
        action_vec = self.embeddings.embed_text(normalized_action)
        action_blob = EmbeddingEngine.vec_to_blob(action_vec) if action_vec else None

        matched_anti_mem: AntiMemory | None = None
        match_source = "exact"
        matched_sources: list[str] = []
        confidence: float = 0.0

        # Tier 1: Fast Substring Containment & Regex on candidate rows
        for r in rows:
            anti_mem = self._row_to_anti_memory(r)
            pattern = anti_mem.pattern

            pat_lower = pattern.lower()
            if pat_lower in combined_text or pat_lower in normalized_action.lower():
                matched_anti_mem = anti_mem
                match_source = "literal_substring"
                matched_sources.append("literal_substring")
                confidence = max(confidence, 1.0)
                break

            # Fallback legacy regex check for explicit pattern expressions
            if any(c in REGEX_CHARS for c in pattern):
                compiled = self._get_or_compile_regex(pattern)
                if compiled is not None:
                    try:
                        if compiled.search(normalized_action):
                            matched_anti_mem = anti_mem
                            match_source = "regex"
                            matched_sources.append("regex")
                            confidence = max(confidence, 1.0)
                            break
                    except Exception:
                        pass

        # Tier 2: Neural Dense Vector Semantic Matching
        if matched_anti_mem is None:
            if self.db.vec_available and action_blob:
                knn_query = """
                    SELECT v.anti_memory_id, v.distance, a.*
                    FROM vec_anti_memories v
                    JOIN anti_memories a ON a.id = v.anti_memory_id
                    WHERE v.embedding MATCH ? AND v.k = 5
                    ORDER BY v.distance
                """
                knn_rows = conn.execute(knn_query, (action_blob,)).fetchall()
                for kr in knn_rows:
                    anti_mem = self._row_to_anti_memory(kr)
                    if not anti_mem.is_active:
                        continue
                    if anti_mem.expires_at is not None and anti_mem.expires_at <= now:
                        continue
                    if project_root and anti_mem.project_root and anti_mem.project_root != project_root.strip():
                        continue

                    sim = 1.0 - float(kr["distance"])
                    effective_threshold = (
                        _READ_ONLY_THRESHOLD
                        if _is_read_only_intent(normalized_action)
                        else self.semantic_threshold
                    )
                    if sim >= effective_threshold:
                        matched_anti_mem = anti_mem
                        match_source = f"semantic_similarity ({sim:.2f})"
                        matched_sources.append(match_source)
                        confidence = max(confidence, round(float(sim), 4))
                        break
            elif action_vec:
                # In-memory fallback
                for r in rows:
                    anti_mem = self._row_to_anti_memory(r)
                    if anti_mem.embedding:
                        sim = EmbeddingEngine.cosine_similarity(action_vec, anti_mem.embedding)
                        effective_threshold = (
                            _READ_ONLY_THRESHOLD
                            if _is_read_only_intent(normalized_action)
                            else self.semantic_threshold
                        )
                        if sim >= effective_threshold:
                            matched_anti_mem = anti_mem
                            match_source = f"semantic_similarity ({sim:.2f})"
                            matched_sources.append(match_source)
                            confidence = max(confidence, round(float(sim), 4))
                            break

        if matched_anti_mem is not None:
            anti_mem = matched_anti_mem
            # Tier 3: Optional LLM Arbiter
            if self.llm_evaluator is not None:
                should_block, llm_reason = self.llm_evaluator(normalized_action, anti_mem)
                if not should_block:
                    logger.info("LLM Arbiter overrode block for '%s' against action '%s'", anti_mem.signature, normalized_action)
                    return InhibitionResult(passed=True)
                if llm_reason:
                    anti_mem.reason = llm_reason

            now = datetime.now(timezone.utc).timestamp()
            with conn:
                conn.execute(
                    """
                    UPDATE anti_memories
                    SET times_triggered = times_triggered + 1,
                        last_seen = ?
                    WHERE id = ?
                    """,
                    (now, anti_mem.id)
                )

            logger.warning(
                "Action '%s' blocked by anti-memory '%s' via %s",
                normalized_action, anti_mem.signature, match_source
            )
            return InhibitionResult(
                passed=False,
                matched_signature=anti_mem.signature,
                reason=f"[{match_source.upper()}] {anti_mem.reason}",
                suggested_fix=anti_mem.suggested_fix,
                severity=anti_mem.severity,
                confidence=confidence,
                matched_sources=matched_sources
            )

        return InhibitionResult(passed=True)

    def forget_anti_memory(self, signature_or_id: str, reason: str = "manual") -> bool:
        """Deactivates an anti-memory so it no longer intercepts actions, logging an audit trail."""
        import uuid
        conn = self.db.get_connection()
        key = signature_or_id.strip()
        with conn:
            row = conn.execute("SELECT * FROM anti_memories WHERE signature = ? OR id = ?", (key, key)).fetchone()
            if row:
                audit_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).timestamp()
                conn.execute(
                    """
                    INSERT INTO forgotten_audit (
                        id, item_type, original_id, signature_or_text, reason, category, metadata_json, forgotten_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (audit_id, "anti_memory", row["id"], row["signature"], reason, row["severity"], None, now)
                )
            cur = conn.execute(
                """
                UPDATE anti_memories
                SET is_active = 0
                WHERE signature = ? OR id = ?
                """,
                (key, key)
            )
            matched = cur.rowcount > 0

        self._compiled_patterns.pop(key, None)
        return matched

    deactivate_anti_memory = forget_anti_memory

    def list_all(self, active_only: bool = True, project_root: str | None = None) -> list[AntiMemory]:
        """Lists recorded anti-memories, optionally filtered by active status and project."""
        conn = self.db.get_connection()
        sql = "SELECT * FROM anti_memories WHERE 1=1"
        params = []

        if active_only:
            sql += " AND is_active = 1"
        if project_root:
            sql += " AND (project_root IS NULL OR project_root = ?)"
            params.append(project_root.strip())

        sql += " ORDER BY times_triggered DESC, last_seen DESC"
        rows = conn.execute(sql, params).fetchall()

        return [self._row_to_anti_memory(r) for r in rows]

    def _get_or_compile_regex(self, pattern: str) -> re.Pattern | None:
        """Safely retrieves or compiles a regex with length bounds and ReDoS defense."""
        if pattern in self._compiled_patterns:
            return self._compiled_patterns[pattern]

        if len(pattern) > 300:
            self._compiled_patterns[pattern] = None
            return None

        if not any(c in REGEX_CHARS for c in pattern):
            self._compiled_patterns[pattern] = None
            return None

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._compiled_patterns[pattern] = compiled
            return compiled
        except re.error:
            self._compiled_patterns[pattern] = None
            return None

    @staticmethod
    def _row_to_anti_memory(r) -> AntiMemory:
        r_keys = r.keys() if hasattr(r, "keys") else []
        context_tags = json.loads(r["context_tags"]) if ("context_tags" in r_keys and r["context_tags"]) else []
        project_root = r["project_root"] if "project_root" in r_keys else None
        is_active = bool(r["is_active"]) if "is_active" in r_keys else True
        valence = ValenceType(r["valence"]) if ("valence" in r_keys and r["valence"]) else ValenceType.INHIBITORY
        embedding = EmbeddingEngine.blob_to_vec(r["embedding"]) if ("embedding" in r_keys and r["embedding"]) else None
        expires_at = float(r["expires_at"]) if ("expires_at" in r_keys and r["expires_at"] is not None) else None

        return AntiMemory(
            id=r["id"],
            signature=r["signature"],
            pattern=r["pattern"],
            reason=r["reason"],
            suggested_fix=r["suggested_fix"],
            context_tags=context_tags,
            times_triggered=r["times_triggered"],
            severity=r["severity"],
            created_at=r["created_at"],
            last_seen=r["last_seen"],
            valence=valence,
            project_root=project_root,
            is_active=is_active,
            embedding=embedding,
            expires_at=expires_at
        )
