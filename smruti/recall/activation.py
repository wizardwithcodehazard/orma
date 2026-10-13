"""
activation.py - Cue-driven spreading activation with the fan effect (ACT-R). Entities cued by the
query send activation along their current facts; an entity's activation is divided among all of its
facts, so a hub such as `user` lends each of its many facts only a sliver while a specific entity
lends its few facts a lot.
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


def spread(store: Store, space: str, cues: dict[int, float], max_hops: int, decay: float) -> dict[int, float]:
    """Activation of current facts after max_hops steps from the cues. At step s each frontier entity e
    with activation a gives every incident fact a * decay**s / fan(e); the fact's other endpoint entity
    carries that amount into the next frontier. Entities are visited once, so activation never echoes back."""
    activation: dict[int, float] = defaultdict(float)
    frontier, visited = dict(cues), set(cues)
    for step in range(max_hops):
        if not frontier:
            break
        edges = [(fact_id, subject_id, object_id, {subject_id, object_id} & frontier.keys())
                 for fact_id, subject_id, object_id in store.incident_facts(space, frontier)]
        fan: dict[int, int] = defaultdict(int)
        for *_, sources in edges:
            for entity_id in sources:
                fan[entity_id] += 1
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
    return dict(activation)
