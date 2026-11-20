"""
sleep - Background consolidation, the slow path. Pending records are read in batches: their sentences are
shown to the model as numbered lines (assistant turns as short, uncitable context), the model records states
and events citing line numbers (neural), and the consolidator checks and integrates them (symbolic). Only
the LLM call is expensive, so a batch is one call with a short answer. Failed batches are retried on later
cycles and retired after sleep_max_attempts. Records are never deleted: `rebuild` reads them all again,
for example with a stronger model.
"""

import time
import uuid
from dataclasses import dataclass, field

from orma import clock
from orma.config import SmrutiConfig
from orma.sleep.consolidate import ConsolidationReport, Consolidator
from orma.sleep.extract import JSON_SCHEMA, ExtractionContext, extract, format_line
from orma.store import USER_ENTITY, Store
from orma.text import count_tokens


@dataclass
class SleepReport:
    space: str
    batches: int = 0
    episodes: int = 0
    failed_batches: int = 0
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
        """Spaces with enough pending records, or whose newest pending record has gone idle."""
        now = time.time() if now is None else now
        return [space for space, pending, newest in self.store.pending_stats()
                if pending >= self.config.sleep_after_episodes or now - newest >= self.config.sleep_idle_seconds]

    def run(self, space: str, now=None) -> SleepReport:
        now = time.time() if now is None else now
        report = SleepReport(space)
        if not self.store.acquire_sleep_lease(space, self.holder, self.config.sleep_lease_seconds, now):
            return report  # another sleeper owns this space
        try:
            self._consolidate(space, now, report)
        finally:
            self.store.release_sleep_lease(space, self.holder)
        return report

    def _consolidate(self, space: str, now: float, report: SleepReport) -> None:
        cfg = self.config
        tz = clock.zone(cfg.timezone)
        while pending := self.store.pending_episodes(space, cfg.sleep_batch_episodes):
            if not self.store.acquire_sleep_lease(space, self.holder, cfg.sleep_lease_seconds, now):
                return  # the lease lapsed during a long run and someone else took it
            batch, sentences, lines = self._fit_budget(pending, tz)
            ids = [e.id for e in batch]
            if not cfg.enable_semantic or not any(s.role in cfg.citable_roles for s in sentences):
                self.store.mark_consolidated(ids, now)  # nothing to read (or structure is switched off)
                report.episodes += len(ids)
                continue
            try:
                schema = JSON_SCHEMA if cfg.enable_schema_decoding else None
                extraction = extract(self.llm, self.model, lines, self._context(space, sentences), schema=schema)
                consolidation = self.consolidator.apply(space, extraction, sentences, now)
            except Exception as exc:  # noqa: BLE001 - any failure must count against the attempt cap
                self.store.record_sleep_failure(ids, str(exc), cfg.sleep_max_attempts, now)
                report.failed_batches += 1
                break
            report.consolidation.merge(consolidation)
            self.store.mark_consolidated(ids, now)
            report.batches += 1
            report.episodes += len(ids)

    def _fit_budget(self, pending, tz):
        """The longest prefix of pending records whose lines fit sleep_prompt_tokens (always at least one record,
        cut to the budget if it alone is too long), with their sentences and formatted lines."""
        cfg = self.config
        batch, sentences, lines, used = [], [], [], 0
        for episode in pending:
            own = self.store.episode_sentences([episode.id])
            own_lines = [format_line(s, tz, s.role in cfg.citable_roles, cfg.context_chars) for s in own]
            cost = sum(count_tokens(line, cfg.chars_per_token) + 1 for line in own_lines)
            if batch and used + cost > cfg.sleep_prompt_tokens:
                break
            if not batch and cost > cfg.sleep_prompt_tokens:  # one oversized record: keep what fits
                keep, total = 0, 0
                for line in own_lines:
                    total += count_tokens(line, cfg.chars_per_token) + 1
                    if total > cfg.sleep_prompt_tokens:
                        break
                    keep += 1
                own, own_lines = own[:max(keep, 1)], own_lines[:max(keep, 1)]
            batch.append(episode)
            sentences += own
            lines += own_lines
            used += cost
        return batch, sentences, lines

    def _context(self, space: str, sentences) -> ExtractionContext:
        """Known names, attributes and current states near the batch, so the model names things consistently and
        sees what a line may be changing."""
        cfg = self.config
        vec = self.store.embed(" ".join(s.text for s in sentences)[:4000])
        names = []
        for entity, _ in self.store.nearest_entities(space, vec, 20):
            aliases = self.store.aliases(entity.id)
            names.append(f"{entity.name} (also called: {', '.join(aliases)})" if aliases else entity.name)
        user = self.store.find_entity(space, USER_ENTITY)
        if user is not None and not any(n.startswith(USER_ENTITY) for n in names):
            aliases = self.store.aliases(user.id)
            names.insert(0, f"{USER_ENTITY} (also called: {', '.join(aliases)})" if aliases else USER_ENTITY)
        attributes = [a.name for a, _ in self.store.nearest_attributes(space, vec, cfg.sleep_context_attributes)]
        state_ids = [i for i, _ in self.store.vec_search("states", space, vec, cfg.sleep_context_states)]
        states = [st.text for st in self.store.states(state_ids)]
        if user is not None:
            states = [st.text for st in self.store.current_states(space, user.id)] + [
                s for s in states if not s.startswith(f"{USER_ENTITY} ·")]
        return ExtractionContext(names, attributes, list(dict.fromkeys(states))[:cfg.sleep_context_states])
