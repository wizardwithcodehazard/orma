"""
recall - Per-request context assembly with no LLM calls and no writes: fuse lexical and vector
search (each hit weighted by how much evidence it carries: calibrated similarity, query-term
coverage), spread activation from the cued entities (query links, search hits and entities primed by
the last requests) along current facts and to the events they took part in, keep the most relevant
facts and events, rank facts by relevance x base-level activation x importance x confidence (events
without confidence), and pack into the token budget.
"""

import threading
import time
from datetime import datetime
from dataclasses import dataclass

from smruti.config import SmrutiConfig
from smruti.messages import message_text
from smruti.recall.activation import merge_cues, search_cues, spread
from smruti.recall.link import link_entities
from smruti.recall.pack import MemoryItem, Packed, count_tokens, pack
from smruti.recall.rank import base_level, importance_factor, recency, rrf, similarity_weight
from smruti.store import USER_ENTITY, Store

SEPARATOR = "\n\n"
SEARCHED_KINDS = ("facts", "procedures", "summaries")


def top(searched: dict[int, float], activated: dict[int, float], n: int) -> dict[int, float]:
    """Relevance = fused search score + spread activation, for the n most relevant ids."""
    relevance = {i: searched.get(i, 0.0) + activated.get(i, 0.0) for i in searched.keys() | activated.keys()}
    return {i: relevance[i] for i in sorted(relevance, key=lambda i: (-relevance[i], i))[:n]}


@dataclass
class RecallResult:
    messages: list[dict]
    memory_text: str
    tokens: int
    used: list[MemoryItem]


class Recaller:
    """Shared by every request of a process. Holds the only recall state, priming, in RAM: per space,
    entity id -> activation left over from recent requests (short-term, so it resets on restart)."""

    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config
        self._priming: dict[str, dict[int, float]] = {}
        self._priming_lock = threading.Lock()

    def primed(self, space: str) -> dict[int, float]:
        with self._priming_lock:
            return dict(self._priming.get(space, {}))

    def _decay_priming(self, space: str) -> dict[int, float]:
        """One request's worth of decay; returns the space's priming after it."""
        cfg = self.config
        with self._priming_lock:
            decayed = {e: a * cfg.priming_decay for e, a in self._priming.get(space, {}).items()
                       if a * cfg.priming_decay >= cfg.priming_floor}
            self._priming[space] = decayed
            return dict(decayed)

    def _prime(self, space: str, activated: dict[int, float]) -> None:
        """Keeps the priming_size strongest of the existing priming and this recall's entity activations."""
        cfg = self.config
        with self._priming_lock:
            merged = dict(self._priming.get(space, {}))
            for entity_id, a in activated.items():
                merged[entity_id] = max(merged.get(entity_id, 0.0), a)
            strongest = sorted(merged, key=lambda e: (-merged[e], e))[:cfg.priming_size]
            self._priming[space] = {e: merged[e] for e in strongest if merged[e] >= cfg.priming_floor}

    def build(self, space: str, messages: list[dict], budget: int | None = None, now=None) -> RecallResult:
        cfg = self.config
        now = time.time() if now is None else now
        budget = cfg.budget_tokens if budget is None else budget
        system_text = "\n\n".join(message_text(m) for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]
        query = next((message_text(m) for m in reversed(turns) if m.get("role") == "user"), "")

        core = self._core(space)
        core_ids = {item.id for item in core}
        said = {message_text(m).strip() for m in turns}
        priming = self._decay_priming(space) if cfg.enable_priming else {}
        activated: dict[int, float] = {}
        ranked = [item for item in self._ranked(space, query, now, said, priming, activated)
                  if not (item.section == "facts" and item.id in core_ids)]
        if cfg.enable_priming:
            self._prime(space, activated)
        # Reserve the "\n\n" joining system text and memory, then cost the real final content.
        reserved = count_tokens(system_text, cfg.chars_per_token) + (
            count_tokens(SEPARATOR, cfg.chars_per_token) if system_text else 0)
        today = datetime.fromtimestamp(now).date().isoformat()  # local calendar date of the request
        packed = pack(core, ranked, turns, budget - reserved, cfg.chars_per_token, cfg.min_recent_turns, today)
        final = self._assemble(system_text, packed)
        tokens = sum(count_tokens(message_text(m), cfg.chars_per_token) for m in final)
        return RecallResult(final, packed.memory_text, tokens, packed.used)

    def _core(self, space: str) -> list[MemoryItem]:
        if not self.config.enable_core_block:
            return []
        user = self.store.find_entity(space, USER_ENTITY)
        if user is None:
            return []
        facts = [f for f in self.store.current_facts(space, subject_id=user.id)
                 if f.confidence >= self.config.core_min_confidence]
        return [MemoryItem("core", f.id, f.text) for f in sorted(facts, key=lambda f: (f.predicate, f.id))]

    def _ranked(self, space: str, query: str, now: float, said: frozenset[str] = frozenset(),
                priming: dict[int, float] | None = None, activated: dict[int, float] | None = None) -> list[MemoryItem]:
        """priming adds priming_weight x its activation to the cues; activated receives the entity activations
        (cues and what spreading reached) that drove this recall."""
        cfg = self.config
        if not query:
            return []
        vec = self.store.embed(query)
        weight = similarity_weight(self.store.novelty_scores(space), cfg.similarity_floor_percentile,
                                   cfg.novelty_warmup, cfg.enable_similarity_weighting)
        fused = {
            kind: rrf([(i, weight(sim)) for i, sim in self.store.vec_search(kind, space, vec, cfg.search_k)],
                      [(hit.id, hit.query_coverage) for hit in self.store.text_search(kind, space, query, cfg.search_k)],
                      k=cfg.rrf_k)
            for kind in SEARCHED_KINDS + (("events",) if cfg.enable_events else ())
            + (("episodes",) if cfg.enable_episode_recall else ())
        }
        hits = self.store.facts(sorted(fused["facts"]))
        activation: dict[int, float] = {}
        event_activation: dict[int, float] = {}
        if cfg.enable_graph:
            cues = merge_cues(link_entities(self.store, cfg, space, query, vec, weight),
                              search_cues(hits, fused["facts"]))
            for entity_id, a in (priming or {}).items():
                cues[entity_id] = cues.get(entity_id, 0.0) + cfg.priming_weight * a
            activation, event_activation, entities = spread(self.store, space, cues, cfg.max_hops, cfg.hop_decay,
                                                            events=cfg.enable_events)
            if activated is not None:
                activated.update(entities)

        def fresh(timestamp: float) -> float:
            return recency(timestamp, now, cfg.recency_half_life_days)

        def strength(evidence_times: list[float], importance: float, last_used: float) -> float:
            """Base-level activation x importance; the ablation falls back to recency of last use."""
            if not cfg.enable_base_level:
                return fresh(last_used)
            return base_level(evidence_times, now, cfg.base_level_decay) * importance_factor(importance,
                                                                                           cfg.importance_weight)

        # Keep only the most relevant facts and events, so a long tail of faintly activated ones cannot
        # starve procedures and summaries, which pack after them.
        relevance = top(fused["facts"], activation, cfg.max_candidate_facts)
        facts = [(relevance[f.id] * strength(f.evidence_times, f.importance, f.last_used) * f.confidence,
                  MemoryItem("facts", f.id, f.text))
                 for f in self.store.facts(sorted(relevance))]
        # Events have no reinforcement: the one evidence is when sleep learned them (the ablation keeps the old
        # recency of when they happened).
        event_relevance = top(fused.get("events", {}), event_activation, cfg.max_candidate_events)
        events = [(event_relevance[e.id] * strength([e.created_at], e.importance, e.occurred_at),
                   MemoryItem("events", e.id, e.text))
                  for e in self.store.events(sorted(event_relevance))]
        procedures = [(fused["procedures"][p.id] * fresh(p.last_used) * p.confidence,
                       MemoryItem("procedures", p.id, p.text))
                      for p in self.store.procedures(sorted(fused["procedures"]))]
        summaries = [(fused["summaries"][s.id] * fresh(s.updated_at), MemoryItem("summaries", s.id, f"{s.topic}: {s.text}"))
                     for s in self.store.summaries(sorted(fused["summaries"]))]
        # Pending episodes carry what no sleep has consolidated yet; turns already in this request are skipped.
        # Only what the user or a tool said: assistant replies are long and still reach sleep.
        episodes = [(fused["episodes"][e.id] * fresh(e.created_at), MemoryItem("episodes", e.id, self._note(e.text)))
                    for e in self.store.pending_by_ids(space, sorted(fused.get("episodes", ())))
                    if e.role in ("user", "tool") and e.text.strip() not in said]

        ordered: list[MemoryItem] = []
        for group in (facts, events, procedures, summaries, episodes):
            ordered.extend(item for _, item in sorted(group, key=lambda pair: (-pair[0], pair[1].id)))
        return ordered

    def _note(self, text: str) -> str:
        limit = self.config.recall_note_chars
        return text if len(text) <= limit else text[: max(limit - 1, 0)].rstrip() + "…"

    @staticmethod
    def _assemble(system_text: str, packed: Packed) -> list[dict]:
        parts = [p for p in (system_text, packed.memory_text) if p]
        head = [{"role": "system", "content": SEPARATOR.join(parts)}] if parts else []
        return head + packed.turns
