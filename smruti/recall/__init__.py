"""
recall - Per-request context assembly with no LLM calls and no writes: fuse lexical and vector
search, spread activation from the cued entities (query links and search hits) along current
facts, rank by relevance x recency x confidence, and pack into the token budget.
"""

import time
from dataclasses import dataclass

from smruti.config import SmrutiConfig
from smruti.messages import message_text
from smruti.recall.activation import merge_cues, search_cues, spread
from smruti.recall.link import link_entities
from smruti.recall.pack import MemoryItem, Packed, count_tokens, pack
from smruti.recall.rank import recency, rrf
from smruti.store import USER_ENTITY, Store

SEPARATOR = "\n\n"
SEARCHED_KINDS = ("facts", "procedures", "summaries")


@dataclass
class RecallResult:
    messages: list[dict]
    memory_text: str
    tokens: int
    used: list[MemoryItem]


class Recaller:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def build(self, space: str, messages: list[dict], budget: int | None = None, now=None) -> RecallResult:
        cfg = self.config
        now = time.time() if now is None else now
        budget = cfg.budget_tokens if budget is None else budget
        system_text = "\n\n".join(message_text(m) for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]
        query = next((message_text(m) for m in reversed(turns) if m.get("role") == "user"), "")

        core = self._core(space)
        core_ids = {item.id for item in core}
        ranked = [item for item in self._ranked(space, query, now)
                  if not (item.section == "facts" and item.id in core_ids)]
        # Reserve the "\n\n" joining system text and memory, then cost the real final content.
        reserved = count_tokens(system_text, cfg.chars_per_token) + (
            count_tokens(SEPARATOR, cfg.chars_per_token) if system_text else 0)
        packed = pack(core, ranked, turns, budget - reserved, cfg.chars_per_token, cfg.min_recent_turns)
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

    def _ranked(self, space: str, query: str, now: float) -> list[MemoryItem]:
        cfg = self.config
        if not query:
            return []
        vec = self.store.embed(query)
        fused = {
            kind: rrf([i for i, _ in self.store.vec_search(kind, space, vec, cfg.search_k)],
                      [i for i, _ in self.store.text_search(kind, space, query, cfg.search_k)])
            for kind in SEARCHED_KINDS
        }
        hits = self.store.facts(sorted(fused["facts"]))
        activation: dict[int, float] = {}
        if cfg.enable_graph:
            cues = merge_cues(link_entities(self.store, cfg, space, query, vec), search_cues(hits, fused["facts"]))
            activation = spread(self.store, space, cues, cfg.max_hops, cfg.hop_decay)

        def fresh(timestamp: float) -> float:
            return recency(timestamp, now, cfg.recency_half_life_days)

        relevance = {fid: fused["facts"].get(fid, 0.0) + activation.get(fid, 0.0)
                     for fid in fused["facts"].keys() | activation.keys()}
        facts = [(relevance[f.id] * fresh(f.last_used) * f.confidence, MemoryItem("facts", f.id, f.text))
                 for f in self.store.facts(sorted(relevance))]
        procedures = [(fused["procedures"][p.id] * fresh(p.last_used) * p.confidence,
                       MemoryItem("procedures", p.id, p.text))
                      for p in self.store.procedures(sorted(fused["procedures"]))]
        summaries = [(fused["summaries"][s.id] * fresh(s.updated_at), MemoryItem("summaries", s.id, f"{s.topic}: {s.text}"))
                     for s in self.store.summaries(sorted(fused["summaries"]))]

        ordered: list[MemoryItem] = []
        for group in (facts, procedures, summaries):
            ordered.extend(item for _, item in sorted(group, key=lambda pair: (-pair[0], pair[1].id)))
        return ordered

    @staticmethod
    def _assemble(system_text: str, packed: Packed) -> list[dict]:
        parts = [p for p in (system_text, packed.memory_text) if p]
        head = [{"role": "system", "content": SEPARATOR.join(parts)}] if parts else []
        return head + packed.turns
