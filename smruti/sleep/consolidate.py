"""
consolidate.py - Symbolic half of sleep. The model's extraction is checked and integrated by code:

- citation: every memory cites a line of this batch that a user, note, document or tool wrote (an assistant
  line is context, never evidence);
- who it is about (its karta): "user", a claimed user name, or first-person words in the user's own lines
  mean the user; a third-person word ("she") carries over the last person named in the text before it
  (anuvrtti); any other name must occur in the batch or already be known;
- grounding: a state's value must be in the cited line, or paraphrase a word of it ("mother" for "mom");
- time: one value per entity and attribute; a new value closes the old one (kept as history), "ended"
  closes it, a stated earlier date is filed as history rather than overriding a later one, and date words
  are resolved against when the line was said, in the user's timezone.
"""

import time
from dataclasses import dataclass, fields

from smruti import clock
from smruti.clock import parse_time  # noqa: F401 - part of this module's API
from smruti.config import SmrutiConfig
from smruti.sleep.extract import ExtractedEvent, ExtractedState, Extraction
from smruti.store import USER_ENTITY, Sentence, Store
from smruti.store.records import state_text
from smruti.text import made_of, phrases, span_of, words

FIRST_PERSON_ROLES = ("user", "note")  # lines in which "I" is the user


@dataclass
class ConsolidationReport:
    states_added: int = 0
    states_reinforced: int = 0
    states_closed: int = 0
    events_added: int = 0
    uncited: int = 0  # memories citing no citable line of the batch
    unresolved: int = 0  # memories whose holder could not be resolved
    ungrounded: int = 0  # states whose value is not in the line they cite
    aliases_added: int = 0
    aliases_rejected: int = 0

    def merge(self, other: "ConsolidationReport") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


class Consolidator:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def apply(self, space: str, extraction: Extraction, sentences: list[Sentence], now=None) -> ConsolidationReport:
        """Integrates an extraction of the given batch sentences (in the order they were said). Idempotent: applying
        the same extraction again adds nothing."""
        now = time.time() if now is None else now
        report = ConsolidationReport()
        citable = {s.id: s for s in sentences if s.role in self.config.citable_roles}
        user_id = self.store.user_entity(space, now)
        self._user_name(space, extraction.user_name, citable.values(), user_id, report)
        mentioned = [n for item in extraction.states for n in (item.about,)] + [
            n for event in extraction.events for n in event.who]
        resolve = _Holders(self, space, sentences, mentioned, user_id, now)

        position = {s.id: i for i, s in enumerate(sentences)}
        for item in sorted(extraction.states, key=lambda x: position.get(x.s, len(position))):
            sentence = citable.get(item.s)
            if sentence is None:
                report.uncited += 1
                continue
            subject_id = resolve(item.about, sentence)
            if subject_id is None:
                report.unresolved += 1
            elif self.config.enable_grounding and not self._grounded(item.value, sentence):
                report.ungrounded += 1
            else:
                self._state(space, subject_id, item, sentence, report)
        for item in extraction.events:
            sentence = citable.get(item.s)
            if sentence is None:
                report.uncited += 1
                continue
            self._event(space, item, sentence, resolve, report, now)
        return report

    # --- names -----------------------------------------------------------------------

    def _user_name(self, space: str, name: str, citable, user_id: int, report) -> None:
        """A name the user calls themself becomes an alias of the user when one of their own lines contains it and
        it is not already someone else."""
        if not name or name.lower() == USER_ENTITY:
            return
        said = any(s.role in FIRST_PERSON_ROLES and span_of(s.text, name) for s in citable)
        existing = self.store.find_entity(space, name)
        if said and (existing is None or existing.id == user_id):
            self.store.add_alias(space, name, user_id)
            report.aliases_added += 1
        else:
            report.aliases_rejected += 1

    def entity(self, space: str, name: str, now: float) -> int:
        """The entity a name refers to: an exact name or alias, a near-identical known name, or a new entity."""
        exact = self.store.find_entity(space, name)
        if exact is not None:
            self.store.touch_entity(exact.id, now)
            return exact.id
        vec = self.store.embed(name)
        for known, sim in self.store.nearest_entities(space, vec, 5):
            if sim >= self.config.entity_merge_threshold and known.name != USER_ENTITY:
                self.store.touch_entity(known.id, now)
                return known.id
        return self.store.add_entity(space, name, "thing", vec=vec, now=now)

    def _attribute(self, space: str, name: str) -> int:
        """An attribute this memory already uses (by name, or by a near-identical name), else a new one."""
        exact = self.store.find_attribute(space, name)
        if exact is not None:
            return exact.id
        vec = self.store.embed(name.replace("_", " "))
        for known, sim in self.store.nearest_attributes(space, vec, 5):
            if sim >= self.config.attribute_merge_threshold:
                return known.id
        return self.store.add_attribute(space, name, vec=vec)

    # --- checks ----------------------------------------------------------------------

    def _grounded(self, value: str, sentence: Sentence) -> bool:
        """The value's terms occur in the cited line, or it paraphrases a word or word pair of it (by embedding)."""
        value_terms, line_terms = self.store.stems([value, sentence.text])
        if value_terms <= line_terms:
            return True
        grams = phrases(sentence.text)
        if not grams:
            return False
        vecs = self.store.embed_many([value] + grams)
        return float(max(vecs[1:] @ vecs[0])) >= self.config.entity_merge_threshold

    def _same(self, a: str, b: str) -> bool:
        if words(a) == words(b):
            return True
        vecs = self.store.embed_many([a, b])
        return float(vecs[0] @ vecs[1]) >= self.config.entity_merge_threshold

    def _stated(self, phrase: str, sentence: Sentence) -> float | None:
        """When date words in a line say something happened or became true, resolved against when the line was said;
        None when there are none, they do not parse, or they name the day it was said (that is not a stated time)."""
        if not phrase:
            return None
        cfg = self.config
        tz = clock.zone(cfg.timezone)
        stamp = (clock.resolve(phrase, sentence.created_at, tz, cfg.date_order) if cfg.enable_date_phrases
                 else clock.parse_time(phrase, None, tz))
        return None if stamp is None or clock.day(stamp, tz) == clock.day(sentence.created_at, tz) else stamp

    # --- integration -----------------------------------------------------------------

    def _state(self, space: str, subject_id: int, item: ExtractedState, sentence: Sentence, report) -> None:
        attribute_id = self._attribute(space, item.attribute)
        history = self.store.state_history(space, subject_id, attribute_id)
        if any(sentence.id in st.evidence or st.ended_by == sentence.id for st in history):
            return  # this line was already integrated (a retried batch)
        current = self.store.current_states(space, subject_id, attribute_id)
        said_at = sentence.created_at
        since = self._stated(item.since, sentence)
        match = next((st for st in current if self._same(st.value, item.value)), None)
        if item.ended:
            if match is not None:
                self.store.close_state(match.id, max(since or said_at, match.valid_from), ended_by=sentence.id)
                report.states_closed += 1
            return
        if match is not None:
            self.store.reinforce_state(match.id, sentence.id, said_at)
            report.states_reinforced += 1
            return
        start = said_at if since is None else since
        # A value dated before a current value that has its own stated date is history, not a change.
        later = [st.valid_from for st in current if since is not None and st.valid_from_stated and st.valid_from > start]
        subject = self.store.entity(subject_id).name
        vec = self.store.embed(state_text(subject, self.store.attribute(attribute_id).name, item.value))
        with self.store.tx():  # close and insert together: a reader never sees the attribute without a value
            if not later:
                for st in current:
                    self.store.close_state(st.id, max(start, st.valid_from), ended_by=sentence.id)
                    report.states_closed += 1
            self.store.add_state(space, subject_id, attribute_id, item.value, start, sentence.id, said_at,
                                 stated=since is not None, valid_to=min(later) if later else None, vec=vec)
        report.states_added += 1

    def _event(self, space: str, item: ExtractedEvent, sentence: Sentence, resolve, report, now: float) -> None:
        vec = self.store.embed(item.what)
        for known in self.store.events_citing(sentence.id):
            if float(vec @ self.store.embed(known.gist)) >= self.config.entity_merge_threshold:
                return  # already recorded from this line
        participants = {pid for pid in (resolve(name, sentence) for name in item.who) if pid is not None}
        when = self._stated(item.when, sentence)
        self.store.add_event(space, item.what, sentence.created_at if when is None else when, when is not None,
                             sorted(participants), sentence.id, vec=vec, now=now)
        report.events_added += 1


class _Holders:
    """Resolves who a memory is about (its karta) from the name the model gave and the line it cites."""

    def __init__(self, consolidator: Consolidator, space: str, sentences: list[Sentence], mentioned: list[str],
                 user_id: int, now: float):
        cfg = consolidator.config
        self.c, self.store, self.space, self.now = consolidator, consolidator.store, space, now
        self.sentences, self.user_id = sentences, user_id
        self.first_person = {w.casefold() for w in cfg.first_person_words}
        self.pronouns = self.first_person | {w.casefold() for w in cfg.third_person_words}
        self.mentioned = [n for n in dict.fromkeys(mentioned) if not self._is_user(n) and not made_of(n, self.pronouns)]
        self.batch_text = "\n".join(s.text for s in sentences)

    def _is_user(self, name: str) -> bool:
        if name.lower() == USER_ENTITY:
            return True
        found = self.store.find_entity(self.space, name)
        return found is not None and found.id == self.user_id

    def __call__(self, name: str, sentence: Sentence) -> int | None:
        if self._is_user(name):
            return self.user_id
        if made_of(name, self.first_person):
            return self.user_id if sentence.role in FIRST_PERSON_ROLES else None
        if made_of(name, self.pronouns):
            carried = self._carried(sentence)
            return None if carried is None else self.c.entity(self.space, carried, self.now)
        known = self.store.find_entity(self.space, name)
        if known is None and not span_of(self.batch_text, name):
            return None  # a name neither said in this batch nor known: not grounded
        return self.c.entity(self.space, name, self.now)

    def _carried(self, sentence: Sentence) -> str | None:
        """Anuvrtti: the last person or thing named before the reference, in this line or the lines before it."""
        index = next((i for i, s in enumerate(self.sentences) if s.id == sentence.id), len(self.sentences))
        for earlier in reversed(self.sentences[:index + 1]):
            named = [n for n in self.mentioned if span_of(earlier.text, n)]
            named += [e.name for e in self.store.entities_named_in(self.space, earlier.text)
                      if e.id != self.user_id and not made_of(e.name, self.pronouns)]
            if named:
                return max(named, key=lambda n: (span_of(earlier.text, n) or (0, 0))[0])  # the last one named
        return None
