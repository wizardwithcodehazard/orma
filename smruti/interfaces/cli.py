"""
cli.py - Typer Command Line Interface for smruti.
Provides commands for inspection, manual sleep cycles, memory retraction (forget), and FastMCP server.
Engineered for reliable cross-platform terminal output (including Windows cp1252 encodings).
"""

import sys

import typer

from smruti.config import get_config
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.storage.db import get_db

app = typer.Typer(
    name="smruti",
    help="smruti: Biologically Inspired Cognitive Memory Engine for Autonomous AI Agents.",
    add_completion=False
)

def _safe_echo(text: str) -> None:
    """Safely outputs text, encoding fallbacks for Windows cp1252 terminals."""
    try:
        typer.echo(text)
    except UnicodeEncodeError:
        # Fallback to ASCII representation
        encoding = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        typer.echo(safe_text)

@app.command()
def init():
    """Initializes a local .smruti/ memory workspace in the current directory."""
    config = get_config()
    get_db(config)
    _safe_echo(f"Initialized smruti memory database at: {config.db_path}")

@app.command()
def status():
    """Displays active memory statistics: episodes, positive rules, graph edges, and anti-memories."""
    config = get_config()
    db = get_db(config)
    stream = StreamBuffer(db)
    inhibitory = InhibitoryGate(db)
    cortex = Cortex(db, config)

    ep_count = stream.count()
    unconsolidated_count = len(stream.get_unconsolidated())
    rule_count = cortex.count()
    edges_count = len(cortex.list_edges())
    anti_memories = inhibitory.list_all(active_only=False)
    active_anti_count = sum(1 for a in anti_memories if a.is_active)

    _safe_echo("smruti Memory System Status")
    _safe_echo(f"  * Workspace: {config.smruti_dir}")
    _safe_echo(f"  * Tier 1 Episodic Buffer: {ep_count} recorded episodes ({unconsolidated_count} pending consolidation)")
    _safe_echo(f"  * Tier 3A Cortical Mesh: {rule_count} active positive rules | {edges_count} associative graph edges")
    _safe_echo(f"  * Tier 3B Inhibitory Mesh: {active_anti_count} active anti-memories ({len(anti_memories)} total)")

@app.command()
def sleep(
    session_id: str | None = typer.Option(None, help="Optional session ID to consolidate"),
    project_root: str | None = typer.Option(None, help="Optional project root to scope anti-memories")
):
    """Runs Tier 2 Memory Consolidation ("The Sleep Cycle") over unconsolidated episodes."""
    config = get_config()
    db = get_db(config)
    stream = StreamBuffer(db)
    inhibitory = InhibitoryGate(db)
    cortex = Cortex(db, config)
    consolidator = Consolidator(db, config, stream, inhibitory, cortex)

    _safe_echo("[smruti] Running memory consolidation cycle...")
    report = consolidator.sleep(session_id=session_id, project_root=project_root)
    _safe_echo(f"  + Processed {report['processed_episodes']} episodes")
    _safe_echo(f"  + Promoted {report['promoted_anti_memories']} new anti-memories")
    _safe_echo(f"  + Promoted {report['promoted_positive_rules']} new positive rules")
    _safe_echo(f"  + Pruned {report['pruned_decayed_rules']} decayed rules")
    _safe_echo(f"  + Cleared {report['pruned_raw_episodes']} historical raw episodes")

@app.command()
def forget(
    target: str = typer.Argument(..., help="Signature or pattern of anti-memory, or rule ID"),
    is_rule: bool = typer.Option(False, "--rule", "-r", help="Set to true if target is a cortical rule ID instead of anti-memory")
):
    """Retracts or deactivates an outdated anti-memory or cortical rule."""
    config = get_config()
    db = get_db(config)
    if is_rule:
        cortex = Cortex(db, config)
        if cortex.forget_rule(target):
            _safe_echo(f"[OK] Successfully deleted cortical rule '{target}'.")
        else:
            _safe_echo(f"[FAILED] No cortical rule found with ID '{target}'.")
    else:
        inhibitory = InhibitoryGate(db)
        if inhibitory.forget_anti_memory(target):
            _safe_echo(f"[OK] Successfully deactivated inhibitory anti-memory '{target}'.")
        else:
            _safe_echo(f"[FAILED] No anti-memory found with signature '{target}'.")

@app.command()
def audit(limit: int = typer.Option(10, help="Number of recent episodes to show")):
    """Displays recent episodic trajectory stream."""
    config = get_config()
    db = get_db(config)
    stream = StreamBuffer(db)
    episodes = stream.get_recent(limit=limit)

    if not episodes:
        _safe_echo("No episodes recorded in Tier 1 buffer yet.")
        return

    _safe_echo(f"Last {len(episodes)} Recorded Episodes:")
    for ep in episodes:
        status_icon = "+" if ep.status.value == "success" else "x"
        cons_mark = "[consolidated]" if ep.consolidated_at else "[pending]"
        _safe_echo(f"  [{status_icon}] ({ep.status.value.upper()}) {ep.action[:50]} -> {ep.result[:50]} {cons_mark}")

@app.command()
def preflight(
    action: str = typer.Argument(..., help="Command or action to check against inhibitory anti-memories"),
    context: str = typer.Option("", help="Optional execution context or intent"),
    project_root: str | None = typer.Option(None, help="Optional project directory scope")
):
    """Checks an action against the Tier 3B inhibitory gate before execution."""
    config = get_config()
    db = get_db(config)
    inhibitory = InhibitoryGate(db)
    res = inhibitory.check_action(action=action, context=context, project_root=project_root)
    if not res.passed:
        _safe_echo("[BLOCKED BY smruti INHIBITORY GATE]")
        _safe_echo(f"  * Signature: {res.matched_signature}")
        _safe_echo(f"  * Severity: {res.severity.upper()}")
        _safe_echo(f"  * Reason: {res.reason}")
        if res.suggested_fix:
            _safe_echo(f"  * Suggested Fix: {res.suggested_fix}")
        sys.exit(1)
    elif res.matched_signature:
        _safe_echo("[WARNING] Matches a known past failure (not blocked)")
        _safe_echo(f"  * Signature: {res.matched_signature}")
        _safe_echo(f"  * Reason: {res.reason}")
    else:
        _safe_echo("[PASSED] No known failure signatures detected.")

@app.command()
def recall(
    query: str = typer.Argument("", help="Search query or keyword for cortical rules"),
    limit: int = typer.Option(5, help="Maximum number of rules to recall")
):
    """Recalls active cortical rules weighted by dense vector similarity and Ebbinghaus strength."""
    config = get_config()
    db = get_db(config)
    cortex = Cortex(db, config)
    recalled = cortex.recall_rules(query=query, limit=limit)
    if not recalled:
        _safe_echo("No relevant cortical rules found in active memory.")
        return

    _safe_echo(f"[smruti RECALLED HEURISTICS] ({len(recalled)} rules):")
    for i, (rule, score) in enumerate(recalled, 1):
        _safe_echo(f"  {i}. [{rule.category}] {rule.rule_text} (strength: {score:.2f}, hits: {rule.access_count})")

@app.command()
def remember(
    fact: str = typer.Argument(..., help="Declarative fact, architectural invariant, or user preference"),
    category: str = typer.Option("general", help="Semantic category (e.g., architecture, preference, security)"),
    confidence: float = typer.Option(1.0, help="Confidence score (0.0 - 1.0)")
):
    """Stores an explicit declarative fact directly into the cortical knowledge mesh."""
    config = get_config()
    db = get_db(config)
    cortex = Cortex(db, config)
    rule = cortex.add_rule(rule_text=fact, category=category, confidence=confidence, base_strength=1.0)
    _safe_echo(f"[OK] Stored declarative fact [{rule.category}]: '{rule.rule_text}' (ID: {rule.id[:8]})")

@app.command()
def ingest(
    target: str = typer.Argument(..., help="Path to a text file or raw text to chunk and ingest"),
    source: str = typer.Option("unknown", help="Source attribution label (e.g., README.md, ADR-001)"),
    category: str = typer.Option("ingested_knowledge", help="Semantic category tag")
):
    """Ingests documentation, architecture notes, or reference text into the cortical mesh."""
    import os
    config = get_config()
    db = get_db(config)
    cortex = Cortex(db, config)

    # If target is an existing file, read contents
    if os.path.isfile(target):
        if source == "unknown":
            source = os.path.basename(target)
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    else:
        content = target

    from smruti.interfaces.mcp_server import _chunk_text
    chunks = _chunk_text(content)
    if not chunks:
        _safe_echo("[FAILED] No content to ingest.")
        return

    stored, reinforced = 0, 0
    for chunk in chunks:
        r = cortex.add_rule(
            rule_text=f"[{source}] {chunk}",
            category=category,
            confidence=0.85,
            base_strength=0.85
        )
        if r.access_count > 1:
            reinforced += 1
        else:
            stored += 1

    _safe_echo(f"[OK] Ingested '{source}': {len(chunks)} chunks processed ({stored} new, {reinforced} reinforced).")

@app.command()
def preference(
    text: str = typer.Argument(..., help="User preference or directive to store across sessions"),
    confidence: float = typer.Option(1.0, help="Confidence weight (0.0 - 1.0)")
):
    """Stores a persistent cross-session user preference or directive."""
    from smruti.framework import smruti
    mem = smruti()
    rule = mem.set_preference(text, confidence=confidence)
    mem.close()
    _safe_echo(f"[OK] Stored user preference: '{rule.rule_text}'")

@app.command(name="anti-fact")
def record_anti_fact_cli(
    fact: str = typer.Argument(..., help="Negative declarative constraint or forbidden anti-pattern (e.g., 'Do not use Redux Toolkit')"),
    confidence: float = typer.Option(1.0, help="Confidence score (0.0 - 1.0)")
):
    """Records an explicit forbidden anti-pattern or negative constraint."""
    from smruti.framework import smruti
    mem = smruti()
    rule = mem.record_anti_fact(fact, confidence=confidence)
    mem.close()
    _safe_echo(f"[OK] Stored forbidden anti-pattern: '{rule.rule_text}' (ID: {rule.id[:8]})")

@app.command()
def context(

    task: str = typer.Argument("", help="Optional task description or search query for context assembly"),
    limit: int = typer.Option(5, help="Number of heuristics to include")
):
    """Outputs the complete prompt injection block (preferences, invariants, and constraints)."""
    from smruti.framework import smruti
    mem = smruti()
    ctx = mem.get_prompt_context(query=task, limit=limit)
    mem.close()
    _safe_echo(ctx)

@app.command()
def observe(
    statement: str = typer.Argument(..., help="Conversational statement or developer feedback to observe in flight")
):
    """Passively observes a conversational turn and extracts persistent invariants or anti-facts."""
    from smruti.framework import smruti
    mem = smruti()
    rule = mem.observe(statement)
    mem.close()
    if rule:
        _safe_echo(f"[OK] Observed and learned [{rule.category}]: '{rule.rule_text}' (ID: {rule.id[:8]})")
    else:
        _safe_echo("[OK] Evaluated: transient statement, no permanent invariant learned.")

@app.command(name="export")

def export_memory(
    output: str = typer.Option("smruti_bundle.json", "--output", "-o", help="Target JSON file path for exported bundle"),
    project_root: str | None = typer.Option(None, help="Optional project directory filter")
):
    """Exports active cortical rules, anti-memories, and graph edges into a portable bundle."""
    from smruti.framework import smruti
    mem = smruti()
    path = mem.export_file(output, project_root=project_root)
    mem.close()
    _safe_echo(f"[OK] Exported memory bundle to: {path}")

@app.command(name="import")
def import_memory(
    file_path: str = typer.Argument(..., help="Path to JSON bundle file to import"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing rules with identical IDs")
):
    """Imports memory rules and anti-memories from a shared bundle JSON file."""
    from smruti.framework import smruti
    mem = smruti()
    res = mem.import_file(file_path, overwrite=overwrite)
    mem.close()
    _safe_echo(
        f"[OK] Import complete: {res['imported_rules']} rules, "
        f"{res['imported_anti_memories']} anti-memories, {res['imported_edges']} edges."
    )

@app.command()
def mcp():
    """Runs the FastMCP server over standard I/O for Cursor / Claude Code / Antigravity."""
    from smruti.interfaces.mcp_server import run_server
    run_server()

@app.command()
def hook():
    """
    Claude Code hook handler (PreToolUse / PostToolUse / PostToolUseFailure on Bash).
    Reads the hook JSON from stdin. Never blocks the agent on internal errors.
    """
    import json
    import logging
    from smruti.interfaces.hooks import handle_claude_hook
    logging.disable(logging.WARNING)  # stdout/stderr belong to the hook protocol
    try:
        out = handle_claude_hook(json.load(sys.stdin))
    except Exception:
        return  # fail open
    if out:
        print(json.dumps(out))

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    session_id: str = typer.Option("shell", help="Session label for recorded episodes"),
):
    """Runs a shell command with preflight checking and automatic outcome recording: smruti run -- <cmd>"""
    import os
    import shlex
    from smruti.interfaces.hooks import format_warning, run_command
    command = shlex.join(ctx.args) if len(ctx.args) > 1 else " ".join(ctx.args)
    if not command:
        _safe_echo("Usage: smruti run -- <command>")
        raise typer.Exit(2)
    code, res = run_command(command, session_id=session_id, project_root=os.getcwd())
    if not res.passed:
        _safe_echo(f"[BLOCKED BY smruti] {res.matched_signature}: {res.reason}")
    elif res.matched_signature:
        typer.echo(format_warning(res), err=True)
    raise typer.Exit(code)




if __name__ == "__main__":
    app()
