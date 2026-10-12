"""
replay_eval.py - Replays real agent command histories through smruti and measures whether its
warnings are right.

For every Bash command, in chronological order:
  1. preflight check (would smruti have warned / blocked?)
  2. the real outcome is recorded
  3. a sleep cycle runs every --sleep-every commands (the autonomic default cadence)

A warning is "correct" if the command then actually failed, "wrong" if it then succeeded.
A repeat failure is a failure of a (command, error) pair that already failed earlier.

Usage:
  python benchmarks/replay_eval.py                          # ~/.claude/projects transcripts
  python benchmarks/replay_eval.py path/to/transcripts_dir  # any dir of Claude Code *.jsonl
  python benchmarks/replay_eval.py trace.jsonl              # lines: {"action","result","status","session_id","timestamp"}
Runs against a throwaway database; your real .smruti memory is never touched.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def load_claude_transcripts(root: Path) -> list[dict]:
    """Extracts Bash tool calls and their outcomes from Claude Code transcript JSONL files."""
    episodes = []
    for path in sorted(root.rglob("*.jsonl")):
        pending: dict[str, dict] = {}
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (entry.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    cmd = (block.get("input") or {}).get("command", "").strip()
                    if cmd:
                        pending[block.get("id")] = {
                            "action": cmd,
                            "session_id": entry.get("sessionId") or path.stem,
                            "timestamp": entry.get("timestamp", ""),
                        }
                elif block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    ep = pending.pop(block["tool_use_id"])
                    result = block.get("content")
                    if isinstance(result, list):
                        result = "\n".join(b.get("text", "") for b in result if isinstance(b, dict))
                    ep["result"] = str(result or "")[-2000:]
                    ep["status"] = "failure" if block.get("is_error") else "success"
                    episodes.append(ep)
    episodes.sort(key=lambda e: e["timestamp"])
    return episodes


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay(episodes: list[dict], sleep_every: int) -> dict:
    from smruti.config import smrutiConfig
    from smruti.engine.consolidator import Consolidator
    from smruti.engine.cortex import Cortex
    from smruti.engine.inhibitory import InhibitoryGate
    from smruti.engine.stream import StreamBuffer
    from smruti.models import ActionStatus
    from smruti.storage.db import DatabaseManager

    with tempfile.TemporaryDirectory() as tmp:
        config = smrutiConfig(project_dir=Path(tmp), db_filename="replay.db")
        config.auto_consolidate = False
        db = DatabaseManager(config)
        stream = StreamBuffer(db, config=config, auto_consolidate=False, use_queue=False)
        gate = InhibitoryGate(db)
        cortex = Cortex(db, config)
        consolidator = Consolidator(db=db, config=config, stream=stream, inhibitory=gate, cortex=cortex)

        m = {"commands": 0, "failures": 0, "repeat_failures": 0, "repeat_failures_warned": 0,
             "warnings": 0, "warnings_correct": 0, "warnings_wrong": 0, "blocks": 0, "wrong_examples": []}
        seen_failures: set[tuple[str, str]] = set()

        for i, ep in enumerate(episodes, 1):
            action, failed = ep["action"], ep["status"] != "success"
            res = gate.check_action(action)
            flagged = res.matched_signature is not None

            m["commands"] += 1
            if flagged:
                m["warnings" if res.passed else "blocks"] += 1
                m["warnings_correct" if failed else "warnings_wrong"] += 1
                if not failed and len(m["wrong_examples"]) < 5:
                    m["wrong_examples"].append(action[:100])
            if failed:
                m["failures"] += 1
                key = (Consolidator._normalize_command(action), Consolidator._error_signature(ep.get("result", "")))
                if key in seen_failures:
                    m["repeat_failures"] += 1
                    m["repeat_failures_warned"] += flagged
                seen_failures.add(key)

            stream.append(action=action, result=ep.get("result", ""),
                          status=ActionStatus.FAILURE if failed else ActionStatus.SUCCESS,
                          session_id=ep.get("session_id", "replay"))
            if i % sleep_every == 0:
                consolidator.sleep()

        consolidator.sleep()
        m["learned_rules"] = [r.rule_text for r, _ in cortex.recall_rules(limit=10)]
        m["active_anti_memories"] = len(gate.list_all())
        db.close()
        return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=str(Path.home() / ".claude" / "projects"))
    parser.add_argument("--sleep-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0, help="Replay only the last N commands")
    parser.add_argument("--json", action="store_true", help="Print raw metrics as JSON")
    args = parser.parse_args()

    src = Path(os.path.expanduser(args.source))
    episodes = load_trace(src) if src.is_file() else load_claude_transcripts(src)
    if args.limit:
        episodes = episodes[-args.limit:]
    if not episodes:
        sys.exit(f"No Bash commands found in {src}")

    m = replay(episodes, args.sleep_every)
    if args.json:
        print(json.dumps(m, indent=2))
        return

    def pct(a: int, b: int) -> str:
        return f"{100 * a / b:.0f}%" if b else "n/a"

    flagged = m["warnings"] + m["blocks"]
    print(f"Replayed {m['commands']} commands ({m['failures']} failures, {m['repeat_failures']} repeat failures)")
    print(f"  Flagged: {flagged} ({m['warnings']} warnings, {m['blocks']} blocks)")
    print(f"  Precision (flag was followed by a real failure): {pct(m['warnings_correct'], flagged)}")
    print(f"  Wrong flags (command actually succeeded):        {m['warnings_wrong']}")
    print(f"  Repeat failures caught beforehand:               {pct(m['repeat_failures_warned'], m['repeat_failures'])}")
    print(f"  Active learned anti-memories at end:             {m['active_anti_memories']}")
    for ex in m["wrong_examples"]:
        print(f"    wrong flag: {ex}")
    print(f"  Learned rules ({len(m['learned_rules'])} shown):")
    for rule in m["learned_rules"]:
        print(f"    - {rule[:140]}")


if __name__ == "__main__":
    main()
