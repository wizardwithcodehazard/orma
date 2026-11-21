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
from orma.recall.rank import (
    base_level,
    ebbinghaus_retention,
    hawkes_burst,
    mmr,
    personalized_pagerank,
    rocchio_expand,
    rrf,
    similarity_weight,
    use_weight,
)
from orma.recall.rerank import Reranker
from orma.store import USER_ENTITY, Event, Sentence, State, Store
from orma.text import count_tokens

SEPARATOR = "\n\n"


@dataclass
class SearchResult:
    states: list[State] = field(default_factory=list)  # true now, most relevant first
    passages: list[tuple[Sentence, float]] = field(
        default_factory=list
    )  # (sentence, score), best first
    events: list[tuple[Event, float]] = field(default_factory=list)
    outdated: dict[int, State] = field(
        default_factory=dict
    )  # sentence id -> the state that replaced what it said

    def prompt(self) -> str:
        parts: list[str] = []
        for s in self.states:
            parts.append(f"[{s.subject}.{s.attribute}: {s.value}]")
        for ev, _ in self.events:
            parts.append(f"[EVENT: {ev.occurred_at or ''}: {ev.gist}]")
        for p, _ in self.passages:
            parts.append(f"> {p.text}")
        return "\n".join(parts)


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
        self.reranker = Reranker(enabled=config.enable_rerank, model_name=config.rerank_model)

    # --- search ----------------------------------------------------------------------

    def search(
        self,
        space: str,
        query: str,
        now: float | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> SearchResult:
        cfg = self.config
        now = time.time() if now is None else now
        result = SearchResult()
        if not query.strip():
            result.states = self._user_states(space)
            return result
        vec = self.store.embed(query)
        weight = similarity_weight(
            self.store.novelty_scores(space),
            cfg.similarity_floor_percentile,
            cfg.novelty_warmup,
            cfg.enable_similarity_weighting,
        )

        def fused(kind: str, search_vec=vec) -> dict[int, float]:
            return rrf(
                [
                    (i, weight(sim))
                    for i, sim in self.store.vec_search(kind, space, search_vec, cfg.search_k)
                ],
                [
                    (hit.id, hit.query_coverage)
                    for hit in self.store.text_search(kind, space, query, cfg.search_k)
                ],
                k=cfg.rrf_k,
            )

        def strength(times: list[float]) -> float:
            if not cfg.enable_base_level:
                return 1.0
            act_r = base_level(times, now, cfg.base_level_decay)
            burst = (
                hawkes_burst(times, now, cfg.hawkes_alpha, cfg.hawkes_half_life_seconds)
                if cfg.enable_hawkes
                else 0.0
            )
            retention = (
                ebbinghaus_retention(times, now, cfg.ebbinghaus_stability_days, cfg.ebbinghaus_gain)
                if cfg.enable_ebbinghaus
                else 1.0
            )
            composite = (act_r + burst) * retention
            return use_weight(composite, cfg.use_weight)

        scores = fused("sentences")

        # Rocchio Pseudo-Relevance Feedback (Vector Space Query Expansion for Multi-Hop)
        if cfg.enable_rocchio and scores and vec is not None:
            top_feedback_ids = sorted(scores.keys(), key=lambda sid: -scores[sid])[
                : cfg.rocchio_top_k
            ]
            feedback_map = self.store.sentence_vectors(top_feedback_ids)
            feedback_vecs = [feedback_map[sid] for sid in top_feedback_ids if sid in feedback_map]
            if feedback_vecs:
                expanded_vec = rocchio_expand(
                    vec, feedback_vecs, alpha=cfg.rocchio_alpha, beta=cfg.rocchio_beta
                )
                expanded_scores = fused("sentences", search_vec=expanded_vec)
                combined_scores: dict[int, float] = dict(scores)
                for sid, sc in expanded_scores.items():
                    combined_scores[sid] = combined_scores.get(sid, 0.0) * 0.6 + sc * 0.4
                scores = combined_scores

        passages = [
            (s, scores[s.id] * strength([s.created_at]))
            for s in self.store.sentences(scores)
            if s.role in cfg.recall_roles and s.text.strip() not in exclude
        ]
        sorted_passages = sorted(passages, key=lambda p: (-p[1], p[0].id))

        # Cross-Encoder Neural Reranking (Early interaction full cross-attention)
        if self.reranker.enabled and len(sorted_passages) > 1:
            sorted_passages = self.reranker.rerank(
                query, sorted_passages[: cfg.rerank_top_candidates]
            )

        # Maximal Marginal Relevance (Context diversification against redundancy)
        if cfg.enable_mmr and len(sorted_passages) > 1:
            candidate_sids = [s.id for s, _ in sorted_passages[: cfg.search_k]]
            vec_map = self.store.sentence_vectors(candidate_sids)
            cand_tuples = [(s.id, score) for s, score in sorted_passages[: cfg.search_k]]
            selected_ids = mmr(cand_tuples, vec_map, cfg.max_passages, cfg.mmr_lambda)
            p_by_id = {s.id: (s, score) for s, score in sorted_passages}
            result.passages = [p_by_id[sid] for sid in selected_ids if sid in p_by_id]
        else:
            result.passages = sorted_passages[: cfg.max_passages]
        if not cfg.enable_semantic:
            return result

        scores = fused("events")
        events = [(e, scores[e.id] * strength([e.created_at])) for e in self.store.events(scores)]
        result.events = sorted(events, key=lambda p: (-p[1], p[0].id))[: cfg.max_events]

        # Others' states need lexical evidence (shared words, or their holder named): "Now true" is shown as fact,
        # so a vector-only match, which in a fresh space is mostly noise, is not enough.
        hits = self.store.text_search("states", space, query, cfg.search_k)
        scores = {hit.id: hit.query_coverage for hit in hits if hit.query_coverage > 0}
        matched = sorted(self.store.states(scores), key=lambda st: (-scores[st.id], st.id))[
            : cfg.max_query_states
        ]
        named_entities = self.store.entities_named_in(space, query)
        if cfg.enable_spreading_activation:
            named = self._spread_activation_states(
                space, named_entities, cfg.activation_min_threshold
            )
        else:
            named = [
                st
                for entity in named_entities
                for st in self.store.current_states(space, entity.id)
            ]
        states = {st.id: st for st in self._user_states(space) + matched + named}
        result.states = list(states.values())

        # Grounding bridge: bring supporting evidence sentences from activated states into passages
        state_evidence_ids = []
        for st in result.states[:5]:
            if st.sentence_id:
                state_evidence_ids.append(st.sentence_id)
            if st.evidence:
                state_evidence_ids.extend(st.evidence[:2])
        if state_evidence_ids:
            existing_sids = {s.id for s, _ in result.passages}
            evidence_sentences = self.store.sentences(set(state_evidence_ids))
            for es in evidence_sentences:
                if (
                    es.id not in existing_sids
                    and es.role in cfg.recall_roles
                    and es.text.strip() not in exclude
                ):
                    result.passages.append((es, 0.85))
            result.passages.sort(key=lambda p: (-p[1], p[0].id))
            result.passages = result.passages[: cfg.max_passages]

        result.outdated = self._outdated(space, [s.id for s, _ in result.passages])
        return result

    def _spread_activation_states(
        self, space: str, entities: list, min_activation: float = 0.05, max_hops: int = 2
    ) -> list[State]:
        """Multi-hop Personalized PageRank (PPR) on Memory Association Graph."""
        if not entities:
            return []

        seeds: dict[str, float] = {f"e_{e.id}": 1.0 for e in entities}
        adjacency: dict[str, list[str]] = {}
        state_by_id: dict[int, State] = {}

        # Build bipartite entity-state association graph
        for e in entities:
            cur = self.store.current_states(space, e.id)
            e_key = f"e_{e.id}"
            adjacency[e_key] = [f"s_{st.id}" for st in cur]
            for st in cur:
                state_by_id[st.id] = st
                adjacency.setdefault(f"s_{st.id}", []).append(e_key)

                # Hop 2: Traversal into referenced entities mentioned in state value
                if max_hops >= 2 and st.value:
                    for e2 in self.store.entities_named_in(space, st.value):
                        if e2.id != e.id:
                            e2_key = f"e_{e2.id}"
                            adjacency[f"s_{st.id}"].append(e2_key)
                            adjacency.setdefault(e2_key, []).append(f"s_{st.id}")
                            cur2 = self.store.current_states(space, e2.id)
                            for st2 in cur2:
                                state_by_id[st2.id] = st2
                                adjacency.setdefault(e2_key, []).append(f"s_{st2.id}")
                                adjacency.setdefault(f"s_{st2.id}", []).append(e2_key)

        ppr_scores = personalized_pagerank(seeds, adjacency, alpha=0.85)

        ranked_states: list[tuple[float, State]] = []
        for sid, st_obj in state_by_id.items():
            score = ppr_scores.get(f"s_{sid}", 0.0)
            if score >= min_activation:
                ranked_states.append((score, st_obj))

        ranked_states.sort(key=lambda x: (-x[0], x[1].id))
        return [st for _, st in ranked_states]

    def _user_states(self, space: str) -> list[State]:
        user = self.store.find_entity(space, USER_ENTITY)
        if user is None:
            return []
        return sorted(
            self.store.current_states(space, user.id),
            key=lambda st: (-max(st.evidence_times), st.id),
        )

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

    def build(
        self, space: str, messages: list[dict], budget: int | None = None, now=None
    ) -> RecallResult:
        cfg = self.config
        now = time.time() if now is None else now
        budget = cfg.budget_tokens if budget is None else budget
        tz = clock.zone(cfg.timezone)
        system_text = "\n\n".join(message_text(m) for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]
        query = next((message_text(m) for m in reversed(turns) if m.get("role") == "user"), "")
        found = self.search(space, query, now, frozenset(message_text(m).strip() for m in turns))

        candidates = [
            MemoryItem("states", st.id, self._state_line(st, tz), i)
            for i, st in enumerate(found.states)
        ]
        for sentence, _ in found.passages:
            line = sentence.label(tz)
            if sentence.id in found.outdated:
                line += f" [outdated: now {found.outdated[sentence.id].text}]"
            candidates.append(MemoryItem("passages", sentence.id, line, sentence.created_at))
        candidates += [
            MemoryItem("events", e.id, e.label(tz), e.occurred_at) for e, _ in found.events
        ]

        reserved = count_tokens(system_text, cfg.chars_per_token) + (
            count_tokens(SEPARATOR, cfg.chars_per_token) if system_text else 0
        )
        available = budget - reserved
        packed = pack(
            candidates,
            turns,
            available,
            cfg.chars_per_token,
            cfg.min_recent_turns,
            clock.iso_day(now, tz),
            state_cap=int(available * cfg.state_budget_share),
            sutra=getattr(cfg, "enable_sutra_packing", False),
        )
        final = self._assemble(system_text, packed)
        tokens = sum(count_tokens(message_text(m), cfg.chars_per_token) for m in final)
        return RecallResult(final, packed.memory_text, tokens, packed.used)

    @staticmethod
    def _state_line(state: State, tz) -> str:
        return (
            f"{state.text} (since {clock.iso_day(state.valid_from, tz)})"
            if state.valid_from_stated
            else state.text
        )

    @staticmethod
    def _assemble(system_text: str, packed: Packed) -> list[dict]:
        parts = [p for p in (system_text, packed.memory_text) if p]
        head = [{"role": "system", "content": SEPARATOR.join(parts)}] if parts else []
        return head + packed.turns
