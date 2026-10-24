"""
score.py - Checks a memory against a case's expectations, without an LLM: what memory holds and what search
finds are scored separately from how well a model answers with it, so a failure can be traced to sleep,
to retrieval or to the reader.

Matching is lenient about wording (models name attributes differently: home_city, city, lives_in) and strict
about meaning: whose state it is, whether it is still true, and the day an event happened.
"""

import re
from dataclasses import dataclass, field

from smruti import clock
from smruti.store import USER_ENTITY, State, Store


@dataclass
class Check:
    kind: str  # current | closed | absent | event | alias | finds | question | sleep
    label: str
    passed: bool
    detail: str = ""


@dataclass
class CaseScore:
    case_id: str
    checks: list[Check] = field(default_factory=list)
    states_total: int = 0
    sleep_seconds: float = 0.0

    def of(self, kind: str) -> list[Check]:
        return [c for c in self.checks if c.kind == kind]


def state_matches(state: State, spec: dict) -> bool:
    if "any" in spec:
        return any(state_matches(state, alt) for alt in spec["any"])
    text = f"{state.attribute.replace('_', ' ')} {state.value}".lower()
    return state.subject.lower() in {n.lower() for n in spec["about"]} and all(t.lower() in text for t in spec["terms"])


def label(spec: dict) -> str:
    if "any" in spec:
        return " | ".join(label(alt) for alt in spec["any"])
    return f"{'/'.join(spec['about'])}: {' '.join(spec['terms'])}"


def _show(states: list[State]) -> str:
    return "; ".join(st.text for st in states[:3])


def score_memory(store: Store, space: str, expect: dict, tz) -> list[Check]:
    current = store.current_states(space)
    history = store.state_history(space)
    checks = []
    for spec in expect.get("current", []):
        hits = [st for st in current if state_matches(st, spec)]
        checks.append(Check("current", label(spec), bool(hits), _show(hits)))
    for spec in expect.get("closed", []):
        past = [st for st in history if st.valid_to is not None and state_matches(st, spec)]
        still = [st for st in current if state_matches(st, spec)]
        checks.append(Check("closed", label(spec), bool(past) and not still,
                            f"still current: {_show(still)}" if still else _show(past)))
    for spec in expect.get("absent", []):
        wrong = [st for st in current if state_matches(st, spec)]
        checks.append(Check("absent", label(spec), not wrong, _show(wrong)))
    events = store.space_events(space)
    for spec in expect.get("events", []):
        found = [e for e in events if all(t.lower() in e.gist.lower() for t in spec["terms"])]
        days = [clock.iso_day(e.occurred_at, tz) for e in found]
        checks.append(Check("event", f"{' '.join(spec['terms'])} on {spec['date']}",
                            any(day.startswith(spec["date"]) for day in days),
                            ", ".join(days) if found else "no such event"))
    for name in expect.get("user_aliases", []):
        entity = store.find_entity(space, name)
        checks.append(Check("alias", f"{name} is the user", entity is not None and entity.name == USER_ENTITY,
                            "unknown" if entity is None else f"resolves to {entity.name}"))
    return checks


def finds(result, terms: list[str], top: int = 5) -> tuple[bool, str]:
    """Whether a search result holds every term in one of its top passages or in one current state."""
    texts = [s.text for s, _ in result.passages[:top]] + [st.text for st in result.states]
    hit = next((t for t in texts if all(term.lower() in t.lower() for term in terms)), None)
    return hit is not None, hit or (texts[0] if texts else "nothing found")


def _says(text: str, phrase: str) -> bool:
    """phrase occurs as whole words ("no" is not found in "know")."""
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text) is not None


def answer_passes(answer: str, must: list[list[str]], must_not: list[list[str]]) -> bool:
    """Every must group has an alternative in the answer and no must_not group does."""
    text = " ".join(answer.lower().split())
    return (all(any(_says(text, alt) for alt in group) for group in must)
            and not any(any(_says(text, alt) for alt in group) for group in must_not))
