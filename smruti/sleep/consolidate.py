"""
consolidate.py - Symbolic half of sleep. Resolves extracted names onto the entity graph,
normalizes relations, applies temporal truth maintenance (a new value for a one-valued
relation closes the old fact rather than overwriting it), accumulates evidence, and records dated
events linked to the entities that took part. Before any of that, a symbolic hallucination check: whatever the batch's own log does not contain is dropped.
"""

import re
import time
from collections.abc import Collection
from dataclasses import dataclass, fields
from datetime import datetime, timezone

from smruti.config import SmrutiConfig
from smruti.recall.pack import count_tokens
from smruti.sleep.extract import ExtractedEvent, ExtractedFact, ExtractedProcedure, Extraction
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
    facts_agreed: int = 0  # confirmed by a replay pass: one extra reinforcement (also counted as added or reinforced)
    facts_ungrounded: int = 0  # dropped: not grounded in the batch's log
    events_added: int = 0
    events_ungrounded: int = 0  # dropped: a participant not grounded in the batch's log, or none at all

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


def utc_day(stamp: float):
    return datetime.fromtimestamp(stamp, timezone.utc).date()


def surprise(episodes) -> float:
    """Importance of what a batch yields: the mean novelty its episodes had when encoded (unknown: 0.5)."""
    novelties = [0.5 if e.novelty is None else e.novelty for e in episodes]
    return sum(novelties) / len(novelties) if novelties else 0.5


class Consolidator:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def apply(self, space: str, extraction: Extraction, episodes, now=None,
              agreed: Collection[int] = ()) -> ConsolidationReport:
        """Integrates the grounded part of the extraction. agreed holds indices of extraction.facts that a
        replay pass confirmed; each counts as one extra observation. Idempotent per episode batch:
        re-applying the same extraction never inflates support."""
        now = time.time() if now is None else now
        episode_ids = [e.id for e in episodes]
        default_time = min((e.created_at for e in episodes), default=now)
        evidence_time = max((e.created_at for e in episodes), default=now)  # recency follows the evidence, not the sleep
        importance = surprise(episodes)
        days = {utc_day(e.created_at) for e in episodes}
        touched: set[int] = set()
        report = ConsolidationReport()
        user_names, entities, facts, events = self._grounded(space, extraction, episodes, report)
        user_id = self.store.user_entity(space, now) if user_names else None
        for name in user_names:  # aliases first, so every name below resolves to the user
            if name.lower() != USER_ENTITY:
                self.store.add_alias(space, name, user_id)
        resolved = {name.lower(): self._entity(space, name, type_, now) for name, type_ in entities.items()}

        def start(item) -> float:
            stated = self._stated_time(item[1], days)
            return default_time if stated is None else stated

        for index, fact in sorted(facts, key=start):
            self._fact(space, fact, entities, resolved, episode_ids, default_time, days, now, evidence_time, report,
                       agreed=index in agreed, importance=importance)
        for event in events if self.config.enable_events else []:
            self._event(space, event, entities, resolved, episode_ids, default_time, days, now, evidence_time, report,
                        importance)
        # Procedures come from tool turns; chat alone yields junk "lessons".
        for procedure in extraction.procedures if any(e.role == "tool" for e in episodes) else []:
            self._procedure(space, procedure, episode_ids, touched, evidence_time, report)
        for topic, text in extraction.summaries.items():
            self._summary(space, topic, text, now)
            report.summaries_updated += 1
        return report

    def _grounded(self, space: str, extraction: Extraction, episodes, report):
        """Symbolic hallucination check: the user names, entities, facts and events whose terms all occur in
        the batch's log, compared with the lexical index's own tokenizer and stemmer. A fact needs every term of
        its object and, unless it is the user, of its subject; the user (by name or alias) needs no mention. An
        event needs at least one participant and every participant grounded like a subject; its gist is the
        model's own summary and is not checked.
        Returns (user_names, entities, [(index in extraction.facts, fact)], events)."""
        indexed = list(enumerate(extraction.facts))
        if not self.config.enable_grounding:
            return extraction.user_names, extraction.entities, indexed, extraction.events
        names, entity_names = extraction.user_names, list(extraction.entities)
        texts = ([e.text for e in episodes] + names + entity_names + [n for _, f in indexed for n in (f.subject, f.object)]
                 + [n for event in extraction.events for n in event.participants])
        stems = self.store.stems(texts)
        log = set().union(*stems[:len(episodes)])
        found = iter(stems[len(episodes):])
        user_names = [name for name in names if next(found) <= log]
        known = {name.lower() for name in user_names}
        user = self.store.find_entity(space, USER_ENTITY)

        def is_user(name: str) -> bool:
            if name.lower() == USER_ENTITY or name.lower() in known:
                return True
            entity = self.store.find_entity(space, name) if user is not None else None
            return entity is not None and entity.id == user.id

        entities = {name: extraction.entities[name] for name in entity_names
                    if next(found) <= log or is_user(name)}
        facts = []
        for item in indexed:
            subject_terms, object_terms = next(found), next(found)
            subject, object_ = item[1].subject, item[1].object
            if (subject_terms <= log or is_user(subject)) and (object_terms <= log or is_user(object_)):
                facts.append(item)
            else:
                report.facts_ungrounded += 1
        events = []
        for event in extraction.events:
            grounded = [next(found) <= log or is_user(name) for name in event.participants]
            if grounded and all(grounded):
                events.append(event)
            else:
                report.events_ungrounded += 1
        return user_names, entities, facts, events

    @staticmethod
    def _stated_time(item: ExtractedFact | ExtractedEvent, days) -> float | None:
        """The fact's or event's own time, unless it falls on the same UTC day as one of the batch's episodes:
        that is the log line's date copied over, not a time the user stated."""
        stamp = parse_time(item.time, None)
        return None if stamp is None or utc_day(stamp) in days else stamp

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

    def _fact(self, space, fact: ExtractedFact, types, resolved, episode_ids, default_time, days, now, evidence_time,
              report, agreed: bool = False, importance: float = 0.5) -> int:
        """Adds or reinforces the fact (an agreed fact counts as two observations); returns the id of the
        graph fact it landed on."""
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
        done = next((f for f in self.store.fact_history(space, subject_id=subject_id)
                     if f.predicate_id == predicate_id and same(f) and batch <= set(f.source_episode_ids)), None)
        if done is not None:
            return done.id  # this batch already contributed this fact (retry or duplicate)
        observations = 2 if agreed else 1
        report.facts_agreed += int(agreed)
        match = next((f for f in current if same(f, fuzzy=True)), None)
        if match is not None:
            self.store.reinforce_fact(match.id, episode_ids, evidence_time, count=observations, importance=importance)
            report.facts_reinforced += 1
            return match.id

        stated_time = self._stated_time(fact, days)
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
            fact_id = self.store.add_fact(space, subject_id, predicate_id, object_id, object_value, valid_from,
                                          episode_ids, now=evidence_time, valid_to=valid_to, vec=vec,
                                          valid_from_stated=stated, support=observations, importance=importance)
        report.facts_closed += closed
        report.facts_added += 1
        return fact_id

    def _event(self, space, event: ExtractedEvent, types, resolved, episode_ids, default_time, days, now,
               evidence_time, report, importance: float) -> None:
        """Records the event unless this batch already recorded a near-identical one (a retry, or the same
        event extracted twice)."""
        vec = self.store.embed(event.gist)
        recorded = self.store.batch_events(space, episode_ids)
        if any(float(vec @ self.store.embed(e.gist)) >= self.config.vote_match_threshold for e in recorded):
            return
        participants = [resolved.get(name.lower()) or self._entity(space, name, types.get(name), now)
                        for name in event.participants]
        stated = self._stated_time(event, days)
        self.store.add_event(space, event.gist, default_time if stated is None else stated, stated is not None,
                             participants, episode_ids, vec=vec, now=evidence_time, importance=importance)
        report.events_added += 1

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
