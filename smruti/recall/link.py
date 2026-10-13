"""
link.py - Entity linking: the query is mapped onto known graph entities by calibrated embedding
similarity plus symbolic name containment. The result is a set of activation cues, weighted by how
strongly each entity is evoked; nothing (not even `user`) is linked unconditionally.
"""

from collections.abc import Callable

from smruti.config import SmrutiConfig
from smruti.store import Store


def link_entities(store: Store, config: SmrutiConfig, space: str, query: str, vec,
                  weight: Callable[[float], float]) -> dict[int, float]:
    """{entity id: cue weight}: the calibrated weight w(similarity) for nearby entities (dropped at 0),
    and 1.0 for an entity whose every name term occurs in the query (a lexical mention: "New York" is
    mentioned by "I visited New York" but not by "any new ideas?")."""
    linked: dict[int, float] = {}
    for entity, sim in store.nearest_entities(space, vec, config.entity_link_k):
        w = weight(sim)
        if w > 0.0:
            linked[entity.id] = w
    for hit in store.text_search("entities", space, query, config.entity_link_k):
        if hit.row_coverage >= 1.0:
            linked[hit.id] = 1.0
    return linked
