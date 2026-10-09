"""
llm.py - Agent-Brain & Distillation Interface for smruti.

Applies Occam's Razor:
AI coding agents (Claude, Cursor, Copilot, Antigravity) already possess frontier reasoning.
Smruti avoids external API keys, cloud billing, and HTTP client dependencies.
Instead, it:
1. Structures trajectory failure clusters and causal breakthroughs during sleep.
2. Prompts the host agent's existing brain to synthesize domain invariants via MCP.
3. Provides deterministic, zero-cost heuristic fallbacks when running headless/offline.
4. Accepts optional custom in-memory callable hooks for embedded agent workflows.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from smruti.models import AntiMemory

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Lean distillation client using the host agent's existing intelligence.
    Zero API keys, zero network overhead, zero external dependencies.
    """

    def __init__(
        self,
        summarizer: Callable[[str, list[dict[str, Any]]], str] | None = None
    ):
        self.summarizer = summarizer

    def is_configured(self) -> bool:
        """Returns True if a custom in-memory summarizer hook is injected."""
        return self.summarizer is not None

    def evaluate_action_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        """Evaluates preflight action. Uses injected summarizer if present, else conservative block."""
        if self.summarizer:
            try:
                res = self.summarizer("preflight_check", [{"action": action, "pattern": anti_memory.pattern}])
                return (True, res)
            except Exception as e:
                logger.warning("Custom preflight evaluator failed: %s; falling back to default block", e)
        return (True, anti_memory.reason)

    def summarize_failure_cluster(self, cluster_data: list[dict[str, Any]]) -> str:
        """Summarizes repeated failure trajectory into an invariant reason."""
        if self.summarizer:
            try:
                return self.summarizer("failure_summary", cluster_data).strip()
            except Exception as e:
                logger.warning("Custom summarizer failed: %s; falling back to heuristic", e)

        # Baseline zero-cost heuristic
        first = cluster_data[0]
        action = first.get("action", "")[:40]
        err_snippet = first.get("result", "").strip().split("\n")[-1][:120] if first.get("result") else "Non-zero exit"
        return f"Repeated failures in '{action}'. Error: {err_snippet}"

    def distill_resolution_heuristic(self, transitions: list[dict[str, Any]]) -> str:
        """Synthesizes a causal failure-fix breakthrough into a reusable positive rule."""
        if self.summarizer:
            try:
                return self.summarizer("causal_resolution", transitions).strip()
            except Exception as e:
                logger.warning("Custom resolution distiller failed: %s; falling back to heuristic", e)

        # Baseline zero-cost heuristic
        t = transitions[0]
        return f"When '{t.get('failed_action')}' fails, use '{t.get('fix_action')}' instead."
