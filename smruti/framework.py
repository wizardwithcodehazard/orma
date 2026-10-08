"""
framework.py - High-level Python Developer SDK for the smruti Cognitive Memory Engine.
Provides an ergonomic facade and function guard decorator for autonomous agent frameworks
(LangChain, CrewAI, AutoGen, or custom agent loops).
"""

import functools
import time
from typing import Any, Callable, Optional

from smruti.config import smrutiConfig, get_config
from smruti.engine.consolidator import Consolidator
from smruti.engine.cortex import Cortex
from smruti.engine.inhibitory import InhibitoryGate
from smruti.engine.stream import StreamBuffer
from smruti.models import (
    ActionStatus,
    AntiMemory,
    CorticalRule,
    Episode,
    InhibitionResult,
)
from smruti.storage.db import DatabaseManager, get_db


class smrutiInhibitionError(Exception):
    """Raised when an action is blocked by the smruti inhibitory gate."""

    def __init__(self, result: InhibitionResult):
        self.result = result
        super().__init__(
            f"Action blocked by smruti Inhibitory Gate: {result.reason} "
            f"(Fix: {result.suggested_fix or 'None provided'})"
        )


class smruti:
    """
    Unified client for smruti Cognitive Memory.
    Combines working memory stream, inhibitory preflight gate, cortical rule mesh, and sleep consolidation.
    """

    def __init__(
        self,
        config: Optional[smrutiConfig] = None,
        db: Optional[DatabaseManager] = None
    ):
        self.config = config or get_config()
        self.db = db or get_db(self.config)
        self.stream = StreamBuffer(self.db)
        self.inhibitory = InhibitoryGate(self.db)
        self.cortex = Cortex(self.db, self.config)
        self.consolidator = Consolidator(
            self.db,
            self.config,
            stream=self.stream,
            inhibitory=self.inhibitory,
            cortex=self.cortex
        )

    def preflight(
        self,
        action: str,
        context: str = "",
        project_root: Optional[str] = None
    ) -> InhibitionResult:
        """
        Active Preflight Check: Run before executing any command or modifying code.
        Returns InhibitionResult with .passed, .reason, .suggested_fix, etc.
        """
        return self.inhibitory.check_action(
            action,
            context=context,
            project_root=project_root
        )

    def record(
        self,
        action: str,
        outcome: str = "",
        status: ActionStatus | str = ActionStatus.SUCCESS,
        latency_ms: float = 0.0,
        context: str = ""
    ) -> Episode:
        """
        Appends an action-outcome trace to the Tier 1 working memory buffer (<1ms latency).
        """
        if isinstance(status, str):
            s = status.strip().lower()
            if s == "success":
                act_status = ActionStatus.SUCCESS
            elif s == "failure":
                act_status = ActionStatus.FAILURE
            elif s == "error":
                act_status = ActionStatus.ERROR
            elif s == "intercepted":
                act_status = ActionStatus.INTERCEPTED
            else:
                act_status = ActionStatus.FAILURE
        else:
            act_status = status

        return self.stream.append(
            action=action,
            result=outcome,
            status=act_status,
            context=context,
            latency_ms=latency_ms
        )

    # Developer aliases
    check_action = preflight
    record_episode = record

    def recall(
        self,
        query: str = "",
        limit: int = 5,
        reinforce: bool = True
    ) -> list[tuple[CorticalRule, float]]:
        """
        Recalls top-ranked cortical rules using dense vector similarity,
        associative spreading activation, and Ebbinghaus decay.
        """
        return self.cortex.recall_rules(query=query, limit=limit, reinforce=reinforce)

    def add_rule(
        self,
        rule_text: str,
        category: str = "general",
        confidence: float = 1.0,
        base_strength: float = 1.0
    ) -> CorticalRule:
        """
        Stores a verified positive heuristic or architectural invariant in the cortical rule mesh.
        """
        return self.cortex.add_rule(
            rule_text=rule_text,
            category=category,
            confidence=confidence,
            base_strength=base_strength
        )

    def remember(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 1.0
    ) -> CorticalRule:
        """
        Stores an explicit declarative fact directly into the cortical knowledge mesh.
        (Semantic invariants, architecture rules, user preferences, domain knowledge).
        """
        return self.add_rule(
            rule_text=fact,
            category=category,
            confidence=confidence,
            base_strength=1.0
        )

    def ingest(
        self,
        text: str,
        source: str = "unknown",
        category: str = "ingested_knowledge",
        chunk_size: int = 512,
        overlap: int = 64
    ) -> dict[str, Any]:
        """
        Ingests a document or reference text by chunking, embedding, and storing in the cortical mesh.
        """
        text = text.strip()
        if not text:
            return {"status": "error", "message": "Empty text provided."}

        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
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

        stored, reinforced = 0, 0
        for chunk in chunks:
            rule = self.cortex.add_rule(
                rule_text=f"[{source}] {chunk}",
                category=category,
                confidence=0.85,
                base_strength=0.85
            )
            if rule.access_count > 1:
                reinforced += 1
            else:
                stored += 1

        return {
            "status": "success",
            "source": source,
            "category": category,
            "total_chunks": len(chunks),
            "stored": stored,
            "reinforced": reinforced
        }

    def set_preference(
        self,
        preference: str,
        confidence: float = 1.0
    ) -> CorticalRule:
        """
        Stores or reinforces a persistent user preference or behavioral directive across sessions.
        Examples:
        - "User prefers TypeScript for backend scripts over Python"
        - "Always generate tests with pytest before modifying core logic"
        - "User prefers concise answers without fluff"
        """
        return self.remember(
            fact=preference,
            category="user_preference",
            confidence=confidence
        )

    def get_preferences(self, limit: int = 20) -> list[CorticalRule]:
        """
        Retrieves all active user preferences across sessions.
        """
        rules_with_scores = self.cortex.recall_rules(
            query="",
            category="user_preference",
            limit=limit,
            reinforce=False
        )
        return [rule for rule, _ in rules_with_scores]

    def record_anti_fact(
        self,
        anti_fact: str,
        confidence: float = 1.0
    ) -> CorticalRule:
        """
        Stores an explicit negative declarative constraint (anti-fact / forbidden anti-pattern).
        These steer code generation and architectural decisions away from banned patterns
        (e.g. "Do not use Redux Toolkit; strictly use Zustand", "Never use inline CSS").
        """
        return self.remember(
            fact=anti_fact,
            category="anti_pattern",
            confidence=confidence
        )

    def get_anti_facts(self, limit: int = 20) -> list[CorticalRule]:
        """
        Retrieves active negative declarative constraints (anti-patterns).
        """
        rules_with_scores = self.cortex.recall_rules(
            query="",
            category="anti_pattern",
            limit=limit,
            reinforce=False
        )
        return [rule for rule, _ in rules_with_scores]

    def observe(self, statement: str) -> Optional[CorticalRule]:
        """
        Passive in-flight cognitive observer.
        Analyzes developer statements during chat turns using in-process neural extraction.
        Automatically converts and stores:
        - Invariants & guidelines -> cortical rules
        - Banned libraries & negative patterns -> anti-facts
        - Developer directives -> user preferences
        Returns CorticalRule if an invariant was learned, or None if ordinary task chat.
        """
        from smruti.engine.in_flight import get_in_flight_extractor
        extractor = get_in_flight_extractor()
        extracted = extractor.extract(statement)

        if not extracted.is_memory:
            return None

        if extracted.memory_type == "anti_pattern":
            return self.record_anti_fact(extracted.statement, confidence=extracted.confidence)
        elif extracted.memory_type == "user_preference":
            return self.set_preference(extracted.statement, confidence=extracted.confidence)
        else:
            return self.remember(extracted.statement, category=extracted.category, confidence=extracted.confidence)


    def get_prompt_context(
        self,
        query: str = "",
        limit: int = 5,
        project_root: Optional[str] = None
    ) -> str:
        """
        Builds a structured memory context block ready for injection into an agent's system prompt:
        Combines active user preferences, forbidden anti-patterns, relevant architectural heuristics,
        and execution constraints.
        """
        sections: list[str] = ["[smruti ACTIVE MEMORY CONTEXT]"]

        # 1. User preferences & behavioral directives
        prefs = self.get_preferences(limit=10)
        if prefs:
            sections.append("### User Directives & Preferences:")
            for p in prefs:
                sections.append(f"- {p.rule_text}")

        # 2. Strictly Forbidden Anti-Patterns (Negative Declarative Constraints)
        anti_facts = self.get_anti_facts(limit=10)
        if anti_facts:
            sections.append("### Strictly Forbidden Anti-Patterns (Do NOT Suggest or Implement):")
            for af in anti_facts:
                sections.append(f"- [FORBIDDEN] {af.rule_text}")

        # 3. Query-relevant architectural invariants and heuristics
        if query:
            recalled = self.recall(query=query, limit=limit, reinforce=False)
            relevant_rules = [
                r for r, _ in recalled
                if r.category not in {"user_preference", "anti_pattern"}
            ]
            if relevant_rules:
                sections.append("### Codebase Invariants & Heuristics:")
                for r in relevant_rules:
                    sections.append(f"- [{r.category}] {r.rule_text}")

        # 4. Active inhibitory anti-memories (dead end commands/tools to avoid)
        anti_memories = self.inhibitory.list_all(active_only=True, project_root=project_root)
        if anti_memories:
            sections.append("### Active Execution Constraints (Known Dead Ends):")
            for am in anti_memories[:5]:
                fix_note = f" (Fix: {am.suggested_fix})" if am.suggested_fix else ""
                sections.append(f"- Pattern: '{am.pattern}' | Reason: {am.reason}{fix_note}")

        if len(sections) == 1:
            return "[smruti: No active memory constraints or preferences stored.]"

        return "\n".join(sections)



    def sleep(
        self,
        session_id: Optional[str] = None,
        project_root: Optional[str] = None
    ) -> dict[str, int]:
        """
        Runs the Tier 2 memory consolidation cycle.
        """
        return self.consolidator.sleep(session_id=session_id, project_root=project_root)

    def record_anti_memory(
        self,
        signature: str,
        pattern: str,
        reason: str,
        suggested_fix: Optional[str] = None,
        project_root: Optional[str] = None,
        expires_at: Optional[float] = None,
        expires_in_days: Optional[float] = None
    ) -> AntiMemory:
        """
        Explicitly records a dead end or failure pattern in the inhibitory gate, with optional TTL.
        """
        return self.inhibitory.record_anti_memory(
            signature=signature,
            pattern=pattern,
            reason=reason,
            suggested_fix=suggested_fix,
            project_root=project_root,
            expires_at=expires_at,
            expires_in_days=expires_in_days
        )


    def forget(self, target: str, is_rule: bool = False) -> bool:
        """
        Deactivates an anti-memory (by signature or ID) or deletes a cortical rule (by ID).
        """
        if is_rule:
            return self.cortex.forget_rule(target)
        return self.inhibitory.forget_anti_memory(target)

    def export_bundle(self, project_root: Optional[str] = None) -> dict[str, Any]:
        """
        Exports active cortical rules, anti-memories, and graph edges into a portable dictionary.
        """
        from smruti.storage.sync import TeamMemoryBundle
        return TeamMemoryBundle.export_bundle(self.db, project_root=project_root)

    def import_bundle(self, bundle: dict[str, Any], overwrite: bool = False) -> dict[str, int]:
        """
        Imports memory items from a portable dictionary bundle.
        """
        from smruti.storage.sync import TeamMemoryBundle
        return TeamMemoryBundle.import_bundle(self.db, bundle, overwrite=overwrite)

    def export_file(self, file_path: Any, project_root: Optional[str] = None) -> Any:
        """
        Serializes and saves a memory bundle to a JSON file.
        """
        from smruti.storage.sync import TeamMemoryBundle
        return TeamMemoryBundle.export_file(self.db, file_path, project_root=project_root)

    def import_file(self, file_path: Any, overwrite: bool = False) -> dict[str, int]:
        """
        Loads and imports a memory bundle from a JSON file.
        """
        from smruti.storage.sync import TeamMemoryBundle
        return TeamMemoryBundle.import_file(self.db, file_path, overwrite=overwrite)

    def close(self) -> None:
        """Closes underlying database connections."""
        self.db.close()



    def guard(
        self,
        project_root: Optional[str] = None,
        raise_on_blocked: bool = False
    ):
        """
        Decorator for agent tools and execution functions.
        Automatically intercepts blocked actions before execution, records the outcome,
        and either returns an inhibition dictionary or raises smrutiInhibitionError.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Infer action string
                action_text = ""
                if args and isinstance(args[0], str):
                    action_text = args[0]
                elif "action" in kwargs:
                    action_text = str(kwargs["action"])
                elif "command" in kwargs:
                    action_text = str(kwargs["command"])
                elif "cmd" in kwargs:
                    action_text = str(kwargs["cmd"])
                else:
                    action_text = f"{func.__name__}({args}, {kwargs})"

                preflight = self.preflight(action_text, project_root=project_root)
                if not preflight.passed:
                    self.record(
                        action=action_text,
                        outcome=f"Blocked: {preflight.reason}",
                        status=ActionStatus.INTERCEPTED
                    )
                    if raise_on_blocked:
                        raise smrutiInhibitionError(preflight)
                    return {
                        "status": "blocked",
                        "reason": preflight.reason,
                        "suggested_fix": preflight.suggested_fix,
                        "signature": preflight.matched_signature
                    }

                t0 = time.perf_counter()
                try:
                    res = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    self.record(
                        action=action_text,
                        outcome=str(res)[:500],
                        status=ActionStatus.SUCCESS,
                        latency_ms=elapsed
                    )
                    return res
                except Exception as exc:
                    elapsed = (time.perf_counter() - t0) * 1000
                    self.record(
                        action=action_text,
                        outcome=str(exc)[:500],
                        status=ActionStatus.ERROR,
                        latency_ms=elapsed
                    )
                    raise
            return wrapper
        return decorator

Smruti = smruti
SmrutiInhibitionError = smrutiInhibitionError
