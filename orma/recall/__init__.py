"""
recall - Search and context assembly, the fast path: no LLM call and no writes.

search(query) returns three kinds of memory:
- states that are true now: the user's always, others' when the query shares words with them or names their holder;
- passages: kept sentences found by hybrid search (vector + full text, fused by reciprocal rank, each
  similarity weighted against the space's own noise floor), ranked by match with a bounded boost for
  recent and repeated use (ACT-R base level). A sentence whose state has since changed is marked outdated;
- events found the same way.
build(messages) renders them into a "## Memory" block in the system prompt, under the token budget.
"""

import time
from dataclasses import dataclass, field

from orma import clock
from orma.config import SmrutiConfig
from orma.messages import message_text
from orma.recall.pack import MemoryItem, Packed, pack
from orma.recall.rank import base_level, hawkes_burst, rrf, similarity_weight, use_weight
from orma.store import USER_ENTITY, Event, Sentence, State, Store
from orma.text import count_tokens

SEPARATOR = "\n\n"


@dataclass
class SearchResult:
    states: list[State] = field(default_factory=list)  # true now, most relevant first
    passages: list[tuple[Sentence, float]] = field(default_factory=list)  # (sentence, score), best first
    events: list[tuple[Event, float]] = field(default_factory=list)
    outdated: dict[int, State] = field(default_factory=dict)  # sentence id -> the state that replaced what it said


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

    # --- search ----------------------------------------------------------------------

    def search(self, space: str, query: str, now: float | None = None, exclude: frozenset[str] = frozenset()) \
            -> SearchResult:
        cfg = self.config
        now = time.time() if now is None else now
        result = SearchResult()
        if not query.strip():
            result.states = self._user_states(space)
            return result
        vec = self.store.embed(query)
        weight = similarity_weight(self.store.novelty_scores(space), cfg.similarity_floor_percentile,
                                   cfg.novelty_warmup, cfg.enable_similarity_weighting)

        def fused(kind: str) -> dict[int, float]:
            return rrf([(i, weight(sim)) for i, sim in self.store.vec_search(kind, space, vec, cfg.search_k)],
                       [(hit.id, hit.query_coverage) for hit in self.store.text_search(kind, space, query, cfg.search_k)],
                       k=cfg.rrf_k)

        def strength(times: list[float]) -> float:
            if not cfg.enable_base_level:
                return 1.0
            act_r = base_level(times, now, cfg.base_level_decay)
            burst = hawkes_burst(times, now, cfg.hawkes_alpha, cfg.hawkes_half_life_seconds) if cfg.enable_hawkes else 0.0
            return use_weight(act_r + burst, cfg.use_weight)

        scores = fused("sentences")
        passages = [(s, scores[s.id] * strength([s.created_at])) for s in self.store.sentences(scores)
                    if s.role in cfg.recall_roles and s.text.strip() not in exclude]
        result.passages = sorted(passages, key=lambda p: (-p[1], p[0].id))[:cfg.max_passages]
        if not cfg.enable_semantic:
            return result

        scores = fused("events")
        events = [(e, scores[e.id] * strength([e.created_at])) for e in self.store.events(scores)]
        result.events = sorted(events, key=lambda p: (-p[1], p[0].id))[:cfg.max_events]

        # Others' states need lexical evidence (shared words, or their holder named): "Now true" is shown as fact,
        # so a vector-only match, which in a fresh space is mostly noise, is not enough.
        hits = self.store.text_search("states", space, query, cfg.search_k)
        scores = {hit.id: hit.query_coverage for hit in hits if hit.query_coverage > 0}
        matched = sorted(self.store.states(scores), key=lambda st: (-scores[st.id], st.id))[:cfg.max_query_states]
        named = [st for entity in self.store.entities_named_in(space, query)
                 for st in self.store.current_states(space, entity.id)]
        states = {st.id: st for st in self._user_states(space) + matched + named}
        result.states = list(states.values())
        result.outdated = self._outdated(space, [s.id for s, _ in result.passages])
        return result

    def _user_states(self, space: str) -> list[State]:
        user = self.store.find_entity(space, USER_ENTITY)
        if user is None:
            return []
        return sorted(self.store.current_states(space, user.id), key=lambda st: (-max(st.evidence_times), st.id))

    def _outdated(self, space: str, sentence_ids: list[int]) -> dict[int, State]:
        """For each sentence that stated a value since replaced, the current state that replaced it."""
        out = {}
        for old in self.store.states_citing(sentence_ids):
            if old.valid_to is None:
                continue
            now_true = self.store.current_states(space, old.subject_id, old.attribute_id)
            if now_true:
                out[old.sentence_id] = now_true[0]
        return out

    # --- context ---------------------------------------------------------------------

    def build(self, space: str, messages: list[dict], budget: int | None = None, now=None) -> RecallResult:
        cfg = self.config
        now = time.time() if now is None else now
        budget = cfg.budget_tokens if budget is None else budget
        tz = clock.zone(cfg.timezone)
        system_text = "\n\n".join(message_text(m) for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]
        query = next((message_text(m) for m in reversed(turns) if m.get("role") == "user"), "")
        found = self.search(space, query, now, frozenset(message_text(m).strip() for m in turns))

        candidates = [MemoryItem("states", st.id, self._state_line(st, tz), i) for i, st in enumerate(found.states)]
        for sentence, _ in found.passages:
            line = sentence.label(tz)
            if sentence.id in found.outdated:
                line += f" [outdated: now {found.outdated[sentence.id].text}]"
            candidates.append(MemoryItem("passages", sentence.id, line, sentence.created_at))
        candidates += [MemoryItem("events", e.id, e.label(tz), e.occurred_at) for e, _ in found.events]

        reserved = count_tokens(system_text, cfg.chars_per_token) + (
            count_tokens(SEPARATOR, cfg.chars_per_token) if system_text else 0)
        available = budget - reserved
        packed = pack(candidates, turns, available, cfg.chars_per_token, cfg.min_recent_turns,
                      clock.iso_day(now, tz), state_cap=int(available * cfg.state_budget_share))
        final = self._assemble(system_text, packed)
        tokens = sum(count_tokens(message_text(m), cfg.chars_per_token) for m in final)
        return RecallResult(final, packed.memory_text, tokens, packed.used)

    @staticmethod
    def _state_line(state: State, tz) -> str:
        return f"{state.text} (since {clock.iso_day(state.valid_from, tz)})" if state.valid_from_stated else state.text

    @staticmethod
    def _assemble(system_text: str, packed: Packed) -> list[dict]:
        parts = [p for p in (system_text, packed.memory_text) if p]
        head = [{"role": "system", "content": SEPARATOR.join(parts)}] if parts else []
        return head + packed.turns
