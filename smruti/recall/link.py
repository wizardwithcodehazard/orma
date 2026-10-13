"""
link.py - Neural entity linking: the query is mapped onto known graph entities by embedding
similarity plus exact lexical mentions. The result is a set of activation cues, weighted by how
strongly each entity is evoked; nothing (not even `user`) is linked unconditionally.
"""

from smruti.config import SmrutiConfig
from smruti.store import Store


def link_entities(store: Store, config: SmrutiConfig, space: str, query: str, vec) -> dict[int, float]:
    """{entity id: cue weight}: embedding similarity for close entities, 1.0 for a name match."""
    linked: dict[int, float] = {}
    for entity, sim in store.nearest_entities(space, vec, config.entity_link_k):
        if sim >= config.entity_link_threshold:
            linked[entity.id] = sim
    for entity_id, _ in store.text_search("entities", space, query, config.entity_link_k):
        linked[entity_id] = 1.0
    return linked
