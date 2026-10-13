"""
sleep - Idle-time consolidation. Pending episodes are processed in batches: the LLM extracts
structure (neural), the consolidator integrates it into the graph (symbolic). Failed batches
are retried on later cycles and retired after sleep_max_attempts. Old raw episodes and stale,
uncertain procedures are forgotten.
"""

import time
from dataclasses import dataclass, field

from smruti.config import SmrutiConfig
from smruti.sleep.consolidate import ConsolidationReport, Consolidator
from smruti.sleep.extract import ExtractionContext, extract
from smruti.store import Store

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

    def due_spaces(self, now=None) -> list[str]:
        """Spaces with enough pending episodes, or whose newest pending episode has gone idle."""
        now = time.time() if now is None else now
        return [space for space, pending, newest in self.store.pending_stats()
                if pending >= self.config.sleep_after_episodes or now - newest >= self.config.sleep_idle_seconds]

    def run(self, space: str, now=None) -> SleepReport:
        cfg = self.config
        now = time.time() if now is None else now
        report = SleepReport(space)
        while batch := self.store.pending_episodes(space, cfg.sleep_batch_episodes):
            ids = [e.id for e in batch]
            try:
                extraction = extract(self.llm, self.model, batch, self._context(space, batch),
                                     max_words=int(cfg.summary_max_tokens * WORDS_PER_TOKEN))
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
        return report

    def _context(self, space: str, batch) -> ExtractionContext:
        vec = self.store.embed(" ".join(e.text for e in batch))
        entities = [entity.name for entity, _ in self.store.nearest_entities(space, vec, 20)]
        predicates = [p.name for p in self.store.predicates(space)]
        summary_ids = [i for i, _ in self.store.vec_search("summaries", space, vec, 3)]
        summaries = {s.topic: s.text for s in self.store.summaries(summary_ids)}
        return ExtractionContext(entities, predicates, summaries)
