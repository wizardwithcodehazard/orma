"""
consolidate.py - Symbolic half of sleep. Resolves extracted names onto the entity graph,
normalizes relations, applies temporal truth maintenance (a new value for a one-valued
relation closes the old fact rather than overwriting it), and accumulates evidence.
"""

import re
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone

from smruti.config import SmrutiConfig
from smruti.recall.pack import count_tokens
from smruti.sleep.extract import ExtractedFact, ExtractedProcedure, Extraction
from smruti.store import USER_ENTITY, Store
from smruti.store.records import fact_text


@dataclass
class ConsolidationReport:
    facts_added: int = 0
    facts_reinforced: int = 0
    facts_closed: int = 0
    procedures_added: int = 0
    procedures_reinforced: int = 0
    summaries_updated: int = 0

    def merge(self, other: "ConsolidationReport") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def parse_time(value: str, default: float | None) -> float | None:
    if not value:
        return default
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        value += "-01-01"
    elif re.fullmatch(r"\d{4}-\d{2}", value):
        value += "-01"
    if value[-1:] in ("Z", "z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
        stamp = (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, OverflowError, OSError):
        return default
    return stamp


class Consolidator:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def apply(self, space: str, extraction: Extraction, episodes, now=None) -> ConsolidationReport:
        """Idempotent per episode batch: re-applying the same extraction never inflates support."""
        now = time.time() if now is None else now
        episode_ids = [e.id for e in episodes]
        default_time = min((e.created_at for e in episodes), default=now)
        evidence_time = max((e.created_at for e in episodes), default=now)  # recency follows the evidence, not the sleep
        touched: set[int] = set()
        report = ConsolidationReport()
        user_id = self.store.user_entity(space, now) if extraction.user_names else None
        for name in extraction.user_names:  # aliases first, so every name below resolves to the user
            if name.lower() != USER_ENTITY:
                self.store.add_alias(space, name, user_id)
        resolved = {name.lower(): self._entity(space, name, type_, now) for name, type_ in extraction.entities.items()}
        for fact in sorted(extraction.facts, key=lambda f: parse_time(f.time, default_time)):
            self._fact(space, fact, extraction.entities, resolved, episode_ids, default_time, now, evidence_time, report)
        # Procedures come from tool turns; chat alone yields junk "lessons".
        for procedure in extraction.procedures if any(e.role == "tool" for e in episodes) else []:
            self._procedure(space, procedure, episode_ids, touched, evidence_time, report)
        for topic, text in extraction.summaries.items():
            self._summary(space, topic, text, now)
            report.summaries_updated += 1
        return report

    def _entity(self, space: str, name: str, type_: str | None, now: float) -> int:
        """type_ is None for names the extraction did not declare: then any type may merge."""
        if name.lower() == USER_ENTITY:
            return self.store.user_entity(space, now)
        exact = self.store.find_entity(space, name)
        if exact is not None:
            self.store.touch_entity(exact.id, now)
            return exact.id
        vec = self.store.embed(name)
        for entity, sim in self.store.nearest_entities(space, vec, 5):
            if sim >= self.config.entity_merge_threshold and (type_ is None or entity.type.lower() == type_.lower()):
                self.store.touch_entity(entity.id, now)
                return entity.id
        return self.store.add_entity(space, name, type_ or "thing", vec=vec, now=now)

    def _predicate(self, space: str, name: str, cardinality: str) -> int:
        name = "_".join(name.lower().split())
        for known in self.store.predicates(space):
            if "_".join(known.name.lower().split()) == name:
                return known.id
        vec = self.store.embed(name.replace("_", " "))
        for known, sim in self.store.nearest_predicates(space, vec, 5):
            if sim >= self.config.predicate_merge_threshold and known.cardinality == cardinality:
                return known.id
        return self.store.add_predicate(space, name, cardinality, vec=vec)

    def _object_entity(self, space: str, name: str, resolved: dict[str, int], now: float) -> int | None:
        if name.lower() == USER_ENTITY:
            return self.store.user_entity(space, now)
        if name.lower() in resolved:
            return resolved[name.lower()]
        known = self.store.find_entity(space, name)
        return known.id if known is not None else None

    def _fact(self, space, fact: ExtractedFact, types, resolved, episode_ids, default_time, now, evidence_time,
              report) -> None:
        subject_id = resolved.get(fact.subject.lower()) or self._entity(space, fact.subject, types.get(fact.subject), now)
        object_id = self._object_entity(space, fact.object, resolved, now)
        object_value = None if object_id is not None else fact.object
        predicate_id = self._predicate(space, fact.predicate, fact.cardinality)
        current = self.store.current_facts(space, subject_id=subject_id, predicate_id=predicate_id)

        new_object_vec = None

        def same(existing, fuzzy: bool = False) -> bool:
            nonlocal new_object_vec
            if object_id is not None and existing.object_id == object_id:
                return True
            if existing.object.strip().lower() == fact.object.strip().lower():  # entity/literal flip
                return True
            if not fuzzy:
                return False
            # Neural match for paraphrased values ("peanuts" vs "peanut") so they reinforce, not contradict
            if new_object_vec is None:
                new_object_vec = self.store.embed(fact.object)
            return float(new_object_vec @ self.store.embed(existing.object)) >= self.config.entity_merge_threshold

        batch = set(episode_ids)
        if any(f.predicate_id == predicate_id and same(f) and batch <= set(f.source_episode_ids)
               for f in self.store.fact_history(space, subject_id=subject_id)):
            return  # this batch already contributed this fact (retry or duplicate)
        match = next((f for f in current if same(f, fuzzy=True)), None)
        if match is not None:
            self.store.reinforce_fact(match.id, episode_ids, evidence_time)
            report.facts_reinforced += 1
            return

        stated_time = parse_time(fact.time, None)
        stated = stated_time is not None
        valid_from = default_time if stated_time is None else stated_time
        text = fact_text(self.store.entity(subject_id).name, self.store.predicate(predicate_id).name,
                         self.store.entity(object_id).name if object_id else object_value)
        vec = self.store.embed(text)  # inference before the lock, so the transaction below stays short
        with self.store.tx():  # close + insert together: a reader never sees a one-valued relation without a value
            valid_to = None
            closed = 0
            if self.store.predicate(predicate_id).cardinality == "one":
                # Back-dating only compares two stated times; an inferred time is just when it was first seen.
                later = [old.valid_from for old in current if stated and old.valid_from_stated and old.valid_from > valid_from]
                if later:
                    valid_to = min(later)  # historical: the existing newer fact stays current
                else:
                    for old in current:
                        self.store.close_fact(old.id, max(valid_from, old.valid_from))
                        closed += 1
            self.store.add_fact(space, subject_id, predicate_id, object_id, object_value, valid_from, episode_ids,
                                now=evidence_time, valid_to=valid_to, vec=vec,
                                valid_from_stated=stated)
        report.facts_closed += closed
        report.facts_added += 1

    def _procedure(self, space: str, procedure: ExtractedProcedure, episode_ids, touched: set[int],
                   evidence_time: float, report) -> None:
        vec = self.store.embed(f"When {procedure.trigger}: {procedure.action}")
        nearest = self.store.nearest_procedure(space, vec)
        if nearest is not None and nearest[1] >= self.config.procedure_merge_threshold:
            found = nearest[0]
            # Idempotent per batch: skip when an earlier apply of this batch already counted here (a procedure
            # touched earlier in this same apply may still take further evidence from the batch).
            if found.id not in touched and set(episode_ids) <= set(found.source_episode_ids):
                return
            self.store.add_procedure_evidence(found.id, procedure.works, evidence_time, episode_ids)
            touched.add(found.id)
            report.procedures_reinforced += 1
        else:
            touched.add(self.store.add_procedure(space, procedure.trigger, procedure.action,
                                                 procedure.error_class or None, procedure.works, vec=vec,
                                                 now=evidence_time, source_episode_ids=episode_ids))
            report.procedures_added += 1

    def _summary(self, space: str, topic: str, text: str, now: float) -> None:
        cpt = self.config.chars_per_token
        text = text[: int(self.config.summary_max_tokens * cpt)].strip()
        self.store.upsert_summary(space, topic, text, count_tokens(text, cpt), now)
