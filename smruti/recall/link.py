"""
link.py - Neural entity linking: the query is mapped onto known graph entities by embedding
similarity plus exact lexical mentions. The user entity is always linked.
"""

from smruti.config import SmrutiConfig
from smruti.store import Store


def link_entities(store: Store, config: SmrutiConfig, space: str, query: str, vec) -> list[int]:
    linked = [store.user_entity(space)]
    for entity, sim in store.nearest_entities(space, vec, config.entity_link_k):
        if sim >= config.entity_link_threshold and entity.id not in linked:
            linked.append(entity.id)
    for entity_id, _ in store.text_search("entities", space, query, config.entity_link_k):
        if entity_id not in linked:
            linked.append(entity_id)
    return linked
