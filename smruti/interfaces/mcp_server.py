"""
mcp_server.py - FastMCP Server for smruti Cognitive Memory Engine.
Exposes preflight gate, sub-ms episodic recording, positive heuristics, sleep cycle,
declarative knowledge storage (smruti_remember), and document ingestion (smruti_ingest)
to AI coding environments (Cursor, Claude Code, Antigravity, Windsurf) over stdio.
"""


import math

from fastmcp import FastMCP

from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.models import ActionStatus
from smruti.storage.db import DatabaseManager, get_db

# Initialize FastMCP Server
mcp = FastMCP("smruti Cognitive Memory")

_cached_db: DatabaseManager | None = None
_cached_bundle: tuple[StreamBuffer, InhibitoryGate, Cortex, Consolidator] | None = None

def _get_engine():
    global _cached_db, _cached_bundle
    db = get_db()
    if _cached_bundle is None or _cached_db is not db:
        _cached_db = db
        stream = StreamBuffer(db)
        inhibitory = InhibitoryGate(db)
        cortex = Cortex(db)
        consolidator = Consolidator(db, stream=stream, inhibitory=inhibitory, cortex=cortex)
        _cached_bundle = (stream, inhibitory, cortex, consolidator)
    return _cached_bundle

def _parse_status(status_str: str) -> ActionStatus:
    s = status_str.strip().lower()
    if s == "success":
        return ActionStatus.SUCCESS
    elif s == "failure":
        return ActionStatus.FAILURE
    elif s == "error":
        return ActionStatus.ERROR
    elif s == "intercepted":
        return ActionStatus.INTERCEPTED
    return ActionStatus.FAILURE

@mcp.tool()
def smruti_preflight_check(action: str, context: str = "", project_root: str | None = None) -> str:
    """
    Active Preflight Gate: Call this BEFORE running any terminal command, code modification, or tool.
    Checks against Tier 3B inhibitory anti-memories to prevent known dead ends, recursive loops, and crashes.
    """
    _, inhibitory, _, _ = _get_engine()
    res = inhibitory.check_action(action, context, project_root=project_root)
    if not res.passed:
        sources = ", ".join(res.matched_sources) if res.matched_sources else "unknown"
        return (
            f"[BLOCKED BY smruti INHIBITORY GATE]\n"
            f"Signature: {res.matched_signature}\n"
            f"Severity: {res.severity.upper()}\n"
            f"Confidence: {res.confidence * 100:.1f}%\n"
            f"Matched via: {sources}\n"
            f"Reason: {res.reason}\n"
            f"Suggested Fix: {res.suggested_fix or 'Change parameters to avoid known dead end.'}"
        )
    return "[PASSED] No known failure signatures detected."

@mcp.tool()
def smruti_record_episode(
    action: str,
    outcome: str = "",
    status: str = "success",
    latency_ms: float = 0.0,
    context: str = ""
) -> str:
    """
    Logs an action and its outcome into the Tier 1 episodic stream with sub-millisecond commit latency.
    Zero LLM latency overhead during execution.
    Supports all statuses: success, failure, error, intercepted.
    """
    stream, _, _, _ = _get_engine()
    act_status = _parse_status(status)
    ep = stream.append(
        action=action,
        result=outcome,
        status=act_status,
        context=context,
        latency_ms=latency_ms
    )
    return f"Logged episode {ep.id[:8]} [status={ep.status.value}] in Tier 1 buffer."

@mcp.tool()
def smruti_record_anti_memory(
    signature: str,
    pattern: str,
    reason: str,
    suggested_fix: str = "",
    project_root: str | None = None,
    expires_in_days: float | None = None
) -> str:
    """
    Explicitly stores an inhibitory anti-memory (dead end / fatal bug) to intercept future runs.
    Can be scoped to a specific project_root workspace.
    Set expires_in_days to make the block temporary (e.g. 7.0 for one week). Omit for permanent.
    """
    import datetime as _dt
    _, inhibitory, _, _ = _get_engine()

    # Compute optional TTL expiry timestamp
    expires_at: float | None = None
    if expires_in_days is not None and expires_in_days > 0:
        expires_at = (
            _dt.datetime.now(_dt.timezone.utc).timestamp()
            + expires_in_days * 86400.0
        )

    anti = inhibitory.record_anti_memory(
        signature=signature,
        pattern=pattern,
        reason=reason,
        suggested_fix=suggested_fix or None,
        project_root=project_root
    )

    # Persist expires_at if provided (record_anti_memory doesn't expose it yet)
    if expires_at is not None:
        conn = inhibitory.db.get_connection()
        with conn:
            conn.execute(
                "UPDATE anti_memories SET expires_at = ? WHERE id = ?",
                (expires_at, anti.id)
            )
        scope_str = f" [scoped to {project_root}]" if project_root else " [global]"
        return (
            f"Created inhibitory anti-memory '{anti.signature}' matching pattern '{anti.pattern}'"
            f"{scope_str} [expires in {expires_in_days:.1f} days]."
        )

    scope_str = f" [scoped to {project_root}]" if project_root else " [global]"
    return f"Created inhibitory anti-memory '{anti.signature}' matching pattern '{anti.pattern}'{scope_str} [permanent]."

@mcp.tool()
def smruti_recall_heuristics(query: str = "", limit: int = 5) -> str:
    """
    Recalls top-weighted positive rules (Tier 3A) governed by vector semantic similarity,
    associative graph spreading activation, and biological Ebbinghaus decay.
    Reinforces accessed memories.
    """
    _, _, cortex, _ = _get_engine()
    recalled = cortex.recall_rules(query=query, limit=limit, reinforce=True)
    if not recalled:
        return "No relevant cortical rules found in active memory."

    lines = ["[smruti RECALLED HEURISTICS]"]
    for i, (rule, score) in enumerate(recalled, 1):
        lines.append(
            f"{i}. [{rule.category}] {rule.rule_text} "
            f"(effective_strength: {score:.2f}, access_count: {rule.access_count})"
        )
    return "\n".join(lines)

@mcp.tool()
def smruti_trigger_sleep(project_root: str | None = None) -> str:
    """
    Triggers Tier 2 Memory Consolidation ("The Sleep Cycle"):
    Replays unconsolidated episodes, promotes recurring failures into anti-memories,
    distills verified successes into positive rules, sweeps historical buffers, and decays dormant facts.
    """
    _, _, _, consolidator = _get_engine()
    report = consolidator.sleep(project_root=project_root)
    return (
        f"[CONSOLIDATION CYCLE COMPLETE]\n"
        f"- Episodes Analyzed: {report['processed_episodes']}\n"
        f"- New Anti-Memories Promoted: {report['promoted_anti_memories']}\n"
        f"- Positive Rules Promoted: {report['promoted_positive_rules']}\n"
        f"- Decayed Rules Pruned: {report['pruned_decayed_rules']}\n"
        f"- Historical Episodes Cleared: {report['pruned_raw_episodes']}"
    )

@mcp.tool()
def smruti_forget(target: str, target_type: str = "anti_memory") -> str:
    """
    Explicitly removes or deactivates an outdated anti-memory or cortical rule.
    target_type: 'anti_memory' (signature or ID) or 'rule' (rule ID).
    """
    _, inhibitory, cortex, _ = _get_engine()
    t_type = target_type.strip().lower()
    if t_type in {"anti_memory", "anti", "antimemory"}:
        success = inhibitory.forget_anti_memory(target)
        if success:
            return f"Deactivated inhibitory anti-memory '{target}'."
        return f"No active anti-memory matched '{target}'."
    elif t_type in {"rule", "positive_rule"}:
        success = cortex.forget_rule(target)
        if success:
            return f"Removed cortical rule '{target}'."
        return f"No cortical rule matched ID '{target}'."
    else:
        return f"Unknown target_type '{target_type}'. Use 'anti_memory' or 'rule'."

@mcp.tool()
def smruti_list_anti_memories(project_root: str | None = None) -> str:
    """
    Lists all active inhibitory anti-memories. Helps the agent inspect why an action might be blocked.
    """
    _, inhibitory, _, _ = _get_engine()
    memories = inhibitory.list_all(active_only=True, project_root=project_root)
    if not memories:
        return "No active inhibitory anti-memories found."

    lines = [f"[ACTIVE smruti ANTI-MEMORIES] ({len(memories)} total)"]
    for m in memories:
        scope = f" [project: {m.project_root}]" if m.project_root else " [global]"
        fix = f" | Fix: {m.suggested_fix}" if m.suggested_fix else ""
        lines.append(f"- [{m.signature}] pattern: '{m.pattern}' (triggered: {m.times_triggered}x){scope}{fix}")
    return "\n".join(lines)

@mcp.tool()
def smruti_remember(
    fact: str,
    category: str = "general",
    confidence: float = 1.0
) -> str:
    """
    Stores an explicit declarative fact directly into the cortical knowledge mesh.
    Use this to teach Smruti things that are NOT derived from command execution:
    - Architectural invariants: "The auth service returns 403 for missing scopes, not 401."
    - Team conventions: "Always use React Query v5, never SWR."
    - User preferences: "User prefers flat module structure over nested domains."
    - Domain facts: "The payments table is append-only; never UPDATE or DELETE rows."
    category: semantic grouping (e.g. 'architecture', 'preference', 'security', 'convention').
    """
    _, _, cortex, _ = _get_engine()
    rule = cortex.add_rule(
        rule_text=fact.strip(),
        category=category.strip() or "general",
        confidence=min(1.0, max(0.0, confidence)),
        base_strength=1.0
    )
    return (
        f"Stored declarative fact in cortical mesh: '{rule.rule_text}' "
        f"[category={rule.category}, id={rule.id[:8]}]."
    )


# Sensible limits to prevent massive embedding jobs in a single call
_INGEST_MAX_CHARS = 50_000  # ~12,500 tokens
_INGEST_CHUNK_SIZE = 512    # characters per chunk
_INGEST_CHUNK_OVERLAP = 64  # character overlap between adjacent chunks


def _chunk_text(text: str, chunk_size: int = _INGEST_CHUNK_SIZE, overlap: int = _INGEST_CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping fixed-size character windows.
    Prefers to break at paragraph or sentence boundaries within the window.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # Try to snap to a paragraph boundary inside [end-overlap, end]
        snap_window = text[max(start, end - overlap):end]
        para_idx = snap_window.rfind("\n\n")
        sent_idx = snap_window.rfind(". ")
        if para_idx != -1:
            end = max(start, end - overlap) + para_idx + 2
        elif sent_idx != -1:
            end = max(start, end - overlap) + sent_idx + 2
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < length else length
    return chunks


@mcp.tool()
def smruti_ingest(
    text: str,
    source: str = "unknown",
    category: str = "ingested_knowledge"
) -> str:
    """
    Ingests a block of text (README, architecture doc, API reference, meeting notes, code comments)
    into the cortical knowledge mesh by chunking and embedding each section.
    Agents can then recall this knowledge semantically via smruti_recall_heuristics.
    source: a label for where this text came from (e.g. 'README.md', 'ADR-003', 'Slack thread').
    category: semantic category tag for all chunks (e.g. 'architecture', 'api_docs', 'onboarding').
    """
    _, _, cortex, _ = _get_engine()
    text = text.strip()
    if not text:
        return "[ERROR] Empty text provided. Nothing was ingested."
    if len(text) > _INGEST_MAX_CHARS:
        text = text[:_INGEST_MAX_CHARS]
        truncated = True
    else:
        truncated = False

    chunks = _chunk_text(text)
    if not chunks:
        return "[ERROR] Text chunking produced no content."

    stored = 0
    reinforced = 0
    for chunk in chunks:
        rule_text = f"[{source}] {chunk}"
        rule = cortex.add_rule(
            rule_text=rule_text,
            category=category.strip() or "ingested_knowledge",
            confidence=0.85,
            base_strength=0.85
        )
        # If the rule already existed (semantic dedup), access_count > 0 and was reinforced
        if rule.access_count > 1:
            reinforced += 1
        else:
            stored += 1

    total = stored + reinforced
    trunc_note = f" [Text was truncated at {_INGEST_MAX_CHARS} chars]" if truncated else ""
    return (
        f"[smruti INGEST] Source: '{source}' | "
        f"{total} chunks processed: {stored} new, {reinforced} reinforced existing.{trunc_note}"
    )


@mcp.tool()
def smruti_set_preference(preference: str, confidence: float = 1.0) -> str:
    """
    Records a persistent user preference or behavioral directive across sessions.
    Examples:
    - "User prefers TypeScript for backend scripts over Python"
    - "Always generate tests with pytest before modifying core logic"
    - "User prefers concise answers without fluff"
    """
    _, _, cortex, _ = _get_engine()
    rule = cortex.add_rule(
        rule_text=preference.strip(),
        category="user_preference",
        confidence=min(1.0, max(0.0, confidence)),
        base_strength=1.0
    )
    return f"Recorded persistent user preference: '{rule.rule_text}' (ID: {rule.id[:8]})."


@mcp.tool()
def smruti_get_preferences(limit: int = 20) -> str:
    """
    Lists all active user preferences and behavioral directives.
    Call this at session startup to understand the user's cross-session requirements.
    """
    _, _, cortex, _ = _get_engine()
    recalled = cortex.recall_rules(query="", category="user_preference", limit=limit, reinforce=False)
    if not recalled:
        return "No user preferences recorded yet."
    lines = ["[smruti USER PREFERENCES]"]
    for i, (rule, _) in enumerate(recalled, 1):
        lines.append(f"{i}. {rule.rule_text}")
    return "\n".join(lines)


@mcp.tool()
def smruti_get_prompt_context(task_description: str = "", project_root: str | None = None) -> str:
    """
    Assembles a consolidated memory context block ready for injection into an agent's prompt:
    Aggregates active user preferences, task-relevant architectural invariants, and active anti-memories.
    Call this when planning a task or starting a new session.
    """
    from smruti.framework import smruti
    mem = smruti(db=_get_engine()[0].db if hasattr(_get_engine()[0], "db") else None)
    return mem.get_prompt_context(query=task_description, limit=5, project_root=project_root)


@mcp.tool()
def smruti_export_bundle(project_root: str | None = None) -> str:
    """
    Exports active cortical rules, anti-memories, and graph edges into a portable JSON bundle.
    Useful for sharing repository-level memory across team members or CI workflows.
    """
    import json
    from smruti.framework import smruti
    mem = smruti(db=_get_engine()[0].db if hasattr(_get_engine()[0], "db") else None)
    bundle = mem.export_bundle(project_root=project_root)
    return json.dumps(bundle, indent=2)


@mcp.tool()
def smruti_import_bundle(bundle_json: str, overwrite: bool = False) -> str:
    """
    Imports a team memory bundle JSON string into the local memory database.
    """
    import json
    from smruti.framework import smruti
    mem = smruti(db=_get_engine()[0].db if hasattr(_get_engine()[0], "db") else None)
    try:
        data = json.loads(bundle_json)
        res = mem.import_bundle(data, overwrite=overwrite)
        return (
            f"[smruti BUNDLE IMPORT COMPLETE]\n"

            f"- Rules Imported: {res['imported_rules']}\n"
            f"- Anti-Memories Imported: {res['imported_anti_memories']}\n"
            f"- Graph Edges Imported: {res['imported_edges']}"
        )

    except Exception as e:
        return f"[ERROR] Failed to parse and import memory bundle: {e}"



@mcp.tool()
def smruti_record_anti_fact(anti_fact: str, confidence: float = 1.0) -> str:

    """
    Records an explicit negative declarative constraint (anti-fact / forbidden anti-pattern)
    to steer code generation away from banned packages, patterns, or architecture decisions.
    Examples:
    - "Do not use Redux Toolkit; strictly use Zustand"
    - "Never use Tailwind CSS; use Vanilla CSS modules only"
    - "Do not use synchronous requests in async endpoints"
    """
    from smruti.framework import smruti
    mem = smruti(db=_get_engine()[0].db if hasattr(_get_engine()[0], "db") else None)
    rule = mem.record_anti_fact(anti_fact, confidence=confidence)
    return f"Recorded forbidden anti-pattern: '{rule.rule_text}' (ID: {rule.id[:8]})."


@mcp.tool()
def smruti_observe(user_statement: str) -> str:
    """
    In-flight cognitive observer: Call this with developer chat feedback or instructions.
    Uses in-process neural extraction to automatically detect and store:
    - Architectural rules and codebase standards
    - Forbidden anti-patterns and banned packages
    - User styling and tool preferences
    Gracefully ignores transient task requests without creating unwanted memories.
    """
    from smruti.framework import smruti
    stream, _, _, _ = _get_engine()
    mem = smruti(db=stream.db)
    rule = mem.observe(user_statement)
    if rule:
        return f"[smruti OBSERVED & STORED] [{rule.category}]: '{rule.rule_text}' (ID: {rule.id[:8]})."
    return "[smruti: Statement was transient task context; no permanent invariant detected.]"



# FastMCP Resource: Clients (Cursor, Claude Code, Windsurf) can subscribe to real-time context
@mcp.resource("smruti://active-context")
def smruti_active_context_resource() -> str:
    """
    Real-time active memory context resource providing user preferences,
    forbidden anti-patterns, and architectural invariants without requiring tool calls.
    """
    from smruti.framework import smruti
    stream, _, _, _ = _get_engine()
    mem = smruti(db=stream.db)
    return mem.get_prompt_context(query="")


# FastMCP Prompt: Native prompt template injects memory into system prompts
@mcp.prompt("smruti-context")
def smruti_context_prompt(task_description: str = "") -> str:
    """
    Prompt template injecting Smruti's active memory context into the LLM system prompt.
    """
    from smruti.framework import smruti
    stream, _, _, _ = _get_engine()
    mem = smruti(db=stream.db)
    return mem.get_prompt_context(query=task_description)






def run_server():
    """Runs the FastMCP server over standard I/O."""
    mcp.run()

if __name__ == "__main__":
    run_server()
