"""
recall - Per-request context assembly with no LLM calls and no writes: link the query to
graph entities, expand along current facts, fuse lexical and vector search, rank by
relevance x recency x confidence, and pack into the token budget.
"""

import time
from dataclasses import dataclass

from smruti.config import SmrutiConfig
from smruti.messages import message_text
from smruti.recall.link import link_entities
from smruti.recall.pack import MemoryItem, Packed, count_tokens, pack
from smruti.recall.rank import recency, rrf
from smruti.store import Store

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
        system_cost = count_tokens(system_text, cfg.chars_per_token)
        packed = pack(core, ranked, turns, budget - system_cost, cfg.chars_per_token, cfg.min_recent_turns)
        return RecallResult(self._assemble(system_text, packed), packed.memory_text,
                            packed.tokens + system_cost, packed.used)

    def _core(self, space: str) -> list[MemoryItem]:
        if not self.config.enable_core_block:
            return []
        user_id = self.store.user_entity(space)
        facts = [f for f in self.store.current_facts(space, subject_id=user_id)
                 if f.confidence >= self.config.core_min_confidence]
        return [MemoryItem("core", f.id, f.text) for f in sorted(facts, key=lambda f: (f.predicate, f.id))]

    def _ranked(self, space: str, query: str, now: float) -> list[MemoryItem]:
        cfg = self.config
        vec = self.store.embed(query)
        fused = {
            kind: rrf([i for i, _ in self.store.vec_search(kind, space, vec, cfg.search_k)],
                      [i for i, _ in self.store.text_search(kind, space, query, cfg.search_k)])
            for kind in SEARCHED_KINDS
        }
        graph: dict[int, float] = {}
        if cfg.enable_graph:
            linked = link_entities(self.store, cfg, space, query, vec)
            graph = {fact_id: cfg.hop_decay ** hops
                     for fact_id, hops in self.store.neighborhood(space, linked, cfg.max_hops).items()}

        def fresh(timestamp: float) -> float:
            return recency(timestamp, now, cfg.recency_half_life_days)

        relevance = {fid: max(graph.get(fid, 0.0), fused["facts"].get(fid, 0.0))
                     for fid in graph.keys() | fused["facts"].keys()}
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
        head = [{"role": "system", "content": "\n\n".join(parts)}] if parts else []
        return head + packed.turns
