"""
activation.py - Cue-driven spreading activation with the fan effect (ACT-R). Entities cued by the
query send activation along their current facts and to the events they took part in; an entity's
activation is divided among all of its facts and events, so a hub such as `user` lends each of its many
facts only a sliver while a specific entity lends its few facts a lot.
"""

from collections import defaultdict

from smruti.store import Fact, Store


def search_cues(hits: list[Fact], scores: dict[int, float]) -> dict[int, float]:
    """Entities at either end of a search-hit fact, weighted by the best fused score among their hits."""
    cues: dict[int, float] = {}
    for fact in hits:
        for entity_id in (fact.subject_id, fact.object_id):
            if entity_id is not None:
                cues[entity_id] = max(cues.get(entity_id, 0.0), scores[fact.id])
    return cues


def merge_cues(*cue_sets: dict[int, float]) -> dict[int, float]:
    """Per-entity maximum over several cue sources."""
    merged: dict[int, float] = {}
    for cues in cue_sets:
        for entity_id, weight in cues.items():
            merged[entity_id] = max(merged.get(entity_id, 0.0), weight)
    return merged


def spread(store: Store, space: str, cues: dict[int, float], max_hops: int, decay: float,
           events: bool = True) -> tuple[dict[int, float], dict[int, float]]:
    """Activation of current facts and of events after max_hops steps from the cues, as (facts, events).
    At step s each frontier entity e with activation a gives every incident fact and every event it took
    part in a * decay**s / fan(e), where fan(e) counts both; a fact's other endpoint entity carries that
    amount into the next frontier, while events are leaves. Entities are visited once, so activation never
    echoes back. With events off, the fan counts facts only."""
    activation: dict[int, float] = defaultdict(float)
    event_activation: dict[int, float] = defaultdict(float)
    frontier, visited = dict(cues), set(cues)
    for step in range(max_hops):
        if not frontier:
            break
        edges = [(fact_id, subject_id, object_id, {subject_id, object_id} & frontier.keys())
                 for fact_id, subject_id, object_id in store.incident_facts(space, frontier)]
        taken_part = store.events_for_entities(space, frontier) if events else []
        fan: dict[int, int] = defaultdict(int)
        for *_, sources in edges:
            for entity_id in sources:
                fan[entity_id] += 1
        for _, entity_id in taken_part:
            fan[entity_id] += 1
        for event_id, entity_id in taken_part:
            event_activation[event_id] += frontier[entity_id] * decay ** step / fan[entity_id]
        following: dict[int, float] = defaultdict(float)
        for fact_id, subject_id, object_id, sources in edges:
            for source in sources:
                amount = frontier[source] * decay ** step / fan[source]
                activation[fact_id] += amount
                other = object_id if source == subject_id else subject_id
                if other is not None and other not in visited:
                    following[other] += amount
        visited |= following.keys()
        frontier = following
    return dict(activation), dict(event_activation)
