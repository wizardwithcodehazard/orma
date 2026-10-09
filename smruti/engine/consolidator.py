"""
consolidator.py - Tier 2: The "Sleep" Consolidation & Distillation Engine.
Replays raw episodic trajectories, extracts schemas and heuristics, promotes recurring
errors into permanent inhibitory anti-memories, links causal resolution trajectories,
and sweeps historical raw buffers and decaying cortical rules.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from smruti.config import smrutiConfig, get_config
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.llm import LLMClient
from smruti.engine.stream import StreamBuffer
from smruti.models import ActionStatus, Episode
from smruti.storage.db import DatabaseManager, get_db

logger = logging.getLogger(__name__)

class Consolidator:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        config: smrutiConfig | None = None,
        stream: StreamBuffer | None = None,
        inhibitory: InhibitoryGate | None = None,
        cortex: Cortex | None = None,
        llm_summarizer: Callable[[str, list[dict[str, Any]]], str] | None = None,
        llm_client: LLMClient | None = None
    ):
        self.config = config or get_config()
        self.db = db or get_db(self.config)
        self.stream = stream or StreamBuffer(self.db)
        self.inhibitory = inhibitory or InhibitoryGate(self.db)
        self.cortex = cortex or Cortex(self.db, self.config)
        self.llm_client = llm_client or LLMClient()
        self.llm_summarizer = llm_summarizer

    def sleep(
        self,
        session_id: str | None = None,
        current_time: float | None = None,
        project_root: str | None = None
    ) -> dict[str, Any]:
        """
        Executes memory consolidation ("Sleep Cycle"):
        1. Analyzes unconsolidated episodic trajectories.
        2. Detects recurring failures and promotes them to Inhibitory Anti-Memories.
        3. Discovers multi-step causal resolutions and promotes them to Cortical Rules.
        4. Marks analyzed episodes as consolidated (idempotent execution).
        5. Enforces raw_retention_days cleanup on old consolidated episodes.
        6. Sweeps decaying dormant rules.
        7. Prunes TTL-expired inhibitory anti-memories.
        """
        now = current_time or datetime.now(timezone.utc).timestamp()
        
        # 1. Fetch only unconsolidated episodes
        unconsolidated = self.stream.get_unconsolidated(limit=200, session_id=session_id)
        if not unconsolidated:
            # Still perform maintenance sweeps even if no new episodes
            pruned_rules = self.cortex.prune_decayed_rules(current_time=now)
            pruned_raw = self.stream.prune_older_than(
                self.config.raw_retention_days * 86400.0,
                only_consolidated=True
            )
            pruned_anti = self._prune_expired_anti_memories(now)
            return {
                "processed_episodes": 0,
                "promoted_anti_memories": 0,
                "promoted_positive_rules": 0,
                "pruned_decayed_rules": pruned_rules,
                "pruned_raw_episodes": pruned_raw,
                "pruned_expired_anti_memories": pruned_anti,
                "synthesis_tasks": [],
                "timestamp": now
            }

        episodes = sorted(unconsolidated, key=lambda e: e.timestamp)
        promoted_anti_memories = 0
        promoted_rules = 0
        synthesis_tasks: list[dict[str, Any]] = []

        # 2. Analyze failure repetitions with normalized command & error clustering
        failure_clusters: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            if ep.status in [ActionStatus.FAILURE, ActionStatus.ERROR]:
                cluster_key = self._extract_cluster_key(ep.action)
                failure_clusters[cluster_key].append(ep)

        for cluster_key, cluster in failure_clusters.items():
            if len(cluster) >= 2:
                # Recurring failure pattern detected -> promote to anti-memory
                first_ep = cluster[0]
                sig = f"recurring_failure_{self._slugify(cluster_key)}"
                
                # Check for project scoping in episode metadata or parameter
                ep_proj = project_root or first_ep.metadata.get("project_root")
                
                cluster_data = [
                    {"action": e.action, "result": e.result[:200], "context": e.context}
                    for e in cluster
                ]
                if self.llm_summarizer:
                    try:
                        reason = self.llm_summarizer("failure_summary", cluster_data)
                    except Exception as e:
                        logger.warning("LLM summarizer failed: %s; falling back to heuristic", e)
                        reason = f"Command '{cluster_key}' failed {len(cluster)} times. Sample error: {first_ep.result[:120]}"
                elif self.llm_client.is_configured():
                    reason = self.llm_client.summarize_failure_cluster(cluster_data)
                else:
                    sample_err = first_ep.result.strip().split("\n")[-1][:120] if first_ep.result else "Non-zero exit code"
                    reason = f"Command pattern '{cluster_key}' failed {len(cluster)} times. Error: {sample_err}"

                self.inhibitory.record_anti_memory(
                    signature=sig,
                    pattern=cluster_key,
                    reason=reason,
                    suggested_fix="Check prerequisites and inspect command arguments.",
                    severity="high",
                    project_root=ep_proj
                )
                promoted_anti_memories += 1
                synthesis_tasks.append({
                    "type": "failure_cluster",
                    "pattern": cluster_key,
                    "signature": sig,
                    "count": len(cluster),
                    "sample_error": first_ep.result.strip().split("\n")[-1][:120] if first_ep.result else "Non-zero exit"
                })

        # 3. Multi-Step Causal Discovery (Window of up to 4 episodes)
        # Finds a failure followed by a subsequent resolution in the same session
        processed_pairs = set()
        for i in range(len(episodes)):
            curr_ep = episodes[i]
            if curr_ep.status not in [ActionStatus.FAILURE, ActionStatus.ERROR]:
                continue

            # Look ahead up to 4 steps for a successful resolution
            lookahead_limit = min(len(episodes), i + 5)
            for j in range(i + 1, lookahead_limit):
                succ_ep = episodes[j]
                if succ_ep.status == ActionStatus.SUCCESS and succ_ep.session_id == curr_ep.session_id:
                    pair_key = (curr_ep.action.strip(), succ_ep.action.strip())
                    if pair_key in processed_pairs:
                        break
                    processed_pairs.add(pair_key)

                    # Distill causal heuristic
                    transitions = [{"failed_action": curr_ep.action, "error": curr_ep.result[:150], "fix_action": succ_ep.action}]
                    if self.llm_summarizer:
                        try:
                            rule_text = self.llm_summarizer("causal_resolution", transitions)
                        except Exception:
                            rule_text = f"When '{curr_ep.action}' fails, use '{succ_ep.action}' instead."
                    elif self.llm_client.is_configured():
                        rule_text = self.llm_client.distill_resolution_heuristic(transitions)
                    else:
                        rule_text = f"When '{curr_ep.action}' fails, use '{succ_ep.action}' instead."

                    self.cortex.add_rule(
                        rule_text=rule_text,
                        category="problem_resolution",
                        confidence=0.9,
                        base_strength=0.9,
                        source_episode_ids=[curr_ep.id, succ_ep.id]
                    )
                    promoted_rules += 1
                    synthesis_tasks.append({
                        "type": "causal_resolution",
                        "failed_action": curr_ep.action,
                        "fix_action": succ_ep.action,
                        "error": curr_ep.result[:120]
                    })
                    break  # Found the resolution for curr_ep

        # 4. Mark all processed episodes as consolidated
        ep_ids = [e.id for e in episodes]
        self.stream.mark_consolidated(ep_ids, now)

        # 5. Raw retention cleanup: prune historical episodes older than retention limit
        retention_seconds = self.config.raw_retention_days * 86400.0
        pruned_raw = self.stream.prune_older_than(retention_seconds, only_consolidated=True)

        # 6. Sweep biologically decayed rules
        pruned_rules = self.cortex.prune_decayed_rules(current_time=now)

        # 7. Expire TTL-bound anti-memories (single SQL DELETE, O(1))
        pruned_anti = self._prune_expired_anti_memories(now)

        return {
            "processed_episodes": len(episodes),
            "promoted_anti_memories": promoted_anti_memories,
            "promoted_positive_rules": promoted_rules,
            "pruned_decayed_rules": pruned_rules,
            "pruned_raw_episodes": pruned_raw,
            "pruned_expired_anti_memories": pruned_anti,
            "synthesis_tasks": synthesis_tasks,
            "timestamp": now
        }

    @staticmethod
    def _extract_cluster_key(action: str) -> str:
        """Extracts normalized command or pattern key from action string."""
        tokens = action.strip().split()
        if not tokens:
            return "unknown"
        # If multi-word command like "pip install" or "npm run" or "git push", group top 2 words
        if len(tokens) >= 2 and tokens[0].lower() in {"pip", "npm", "git", "docker", "yarn", "cargo", "pnpm", "poetry", "uv"}:
            return f"{tokens[0]} {tokens[1]}"
        return tokens[0]

    @staticmethod
    def _slugify(text: str) -> str:
        """Creates safe alphanumeric identifier."""
        return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()

    def _prune_expired_anti_memories(self, now: float) -> int:
        """Deletes anti-memories whose expires_at timestamp has passed, logging an audit trail."""
        import uuid
        conn = self.db.get_connection()
        with conn:
            expired_rows = conn.execute(
                "SELECT * FROM anti_memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            ).fetchall()
            for r in expired_rows:
                audit_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO forgotten_audit (
                        id, item_type, original_id, signature_or_text, reason, category, metadata_json, forgotten_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (audit_id, "anti_memory", r["id"], r["signature"], f"ttl_expired (expires_at={r['expires_at']})", r["severity"], None, now)
                )
            cur = conn.execute(
                "DELETE FROM anti_memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
        pruned = cur.rowcount
        if pruned:
            logger.info("Pruned %d TTL-expired anti-memories (logged to forgotten_audit).", pruned)
        return pruned
