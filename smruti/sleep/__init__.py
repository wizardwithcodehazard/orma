"""
sleep - Idle-time consolidation. Pending episodes are processed in batches: the LLM extracts
structure (neural), the consolidator integrates it into the graph (symbolic). Failed batches
are retried on later cycles and retired after sleep_max_attempts. Old raw episodes and stale,
uncertain procedures are forgotten.
"""

import time
import uuid
from dataclasses import dataclass, field

from smruti.config import SmrutiConfig
from smruti.sleep.consolidate import ConsolidationReport, Consolidator
from smruti.recall.pack import count_tokens
from smruti.sleep.extract import ExtractionContext, extract, format_episode
from smruti.store import USER_ENTITY, Store

WORDS_PER_TOKEN = 0.75


@dataclass
class SleepReport:
    space: str
    batches: int = 0
    episodes: int = 0
    failed_batches: int = 0
    pruned_episodes: int = 0
    pruned_procedures: int = 0
    consolidation: ConsolidationReport = field(default_factory=ConsolidationReport)


class Sleeper:
    def __init__(self, store: Store, config: SmrutiConfig, llm, model: str | None = None):
        self.store = store
        self.config = config
        self.llm = llm
        self.model = model
        self.consolidator = Consolidator(store, config)
        self.holder = uuid.uuid4().hex  # identifies this sleeper in the space's lease

    def due_spaces(self, now=None) -> list[str]:
        """Spaces with enough pending episodes, or whose newest pending episode has gone idle."""
        now = time.time() if now is None else now
        return [space for space, pending, newest in self.store.pending_stats()
                if pending >= self.config.sleep_after_episodes or now - newest >= self.config.sleep_idle_seconds]

    def run(self, space: str, now=None) -> SleepReport:
        cfg = self.config
        now = time.time() if now is None else now
        report = SleepReport(space)
        if not self.store.acquire_sleep_lease(space, self.holder, cfg.sleep_lease_seconds, now):
            return report  # another sleeper owns this space
        try:
            self._consolidate(space, now, report)
        finally:
            self.store.release_sleep_lease(space, self.holder)
        return report

    def _consolidate(self, space: str, now: float, report: SleepReport) -> None:
        cfg = self.config
        while pending := self.store.pending_episodes(space, cfg.sleep_batch_episodes):
            batch, max_chars = self._fit_budget(pending)
            if not self.store.acquire_sleep_lease(space, self.holder, cfg.sleep_lease_seconds, now):
                return  # the lease lapsed during a long run and someone else took it
            ids = [e.id for e in batch]
            try:
                extraction = extract(self.llm, self.model, batch, self._context(space, batch),
                                     max_words=int(cfg.summary_max_tokens * WORDS_PER_TOKEN),
                                     max_chars=max_chars)
                consolidation = self.consolidator.apply(space, extraction, batch, now)
            except Exception as exc:  # any failure must count against the attempt cap
                self.store.record_sleep_failure(ids, str(exc), cfg.sleep_max_attempts, now)
                report.failed_batches += 1
                break
            report.consolidation.merge(consolidation)
            self.store.mark_consolidated(ids, now)
            report.batches += 1
            report.episodes += len(ids)
        report.pruned_episodes = self.store.prune_episodes(now - cfg.episode_retention_days * 86400.0)
        report.pruned_procedures = self.store.prune_procedures(now - cfg.procedure_ttl_days * 86400.0,
                                                               cfg.procedure_min_confidence)

    def _fit_budget(self, pending) -> tuple[list, int]:
        """The longest prefix of pending episodes whose formatted log fits sleep_prompt_tokens (always at least
        one), and the per-episode text cap: sleep_max_episode_chars, lowered so a single oversized episode fits."""
        cfg = self.config
        budget_chars = int(cfg.sleep_prompt_tokens * cfg.chars_per_token)
        max_chars = min(cfg.sleep_max_episode_chars, max(0, budget_chars - len(format_episode(pending[0], 0))))
        lines: list[str] = []
        for episode in pending:
            lines.append(format_episode(episode, max_chars))
            if len(lines) > 1 and count_tokens("\n".join(lines), cfg.chars_per_token) > cfg.sleep_prompt_tokens:
                lines.pop()
                break
        return pending[:len(lines)], max_chars

    def _context(self, space: str, batch) -> ExtractionContext:
        cfg = self.config
        vec = self.store.embed(" ".join(e.text for e in batch))
        nearest = [entity for entity, _ in self.store.nearest_entities(space, vec, 20)]
        user = self.store.find_entity(space, USER_ENTITY)
        if user is not None and all(e.id != user.id for e in nearest):
            nearest.insert(0, user)  # the person chatting is always a candidate subject
        entities = []
        for entity in nearest:
            aliases = self.store.aliases(entity.id)
            entities.append(f"{entity.name} (also called: {', '.join(aliases)})" if aliases else entity.name)
        predicates = [p.name for p, _ in self.store.nearest_predicates(space, vec, cfg.sleep_context_predicates)]
        summary_ids = [i for i, _ in self.store.vec_search("summaries", space, vec, 3)]
        summaries = {s.topic: s.text for s in self.store.summaries(summary_ids)}
        return ExtractionContext(entities, predicates, summaries, self.store.summary_topics(space))
