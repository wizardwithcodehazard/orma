"""
encode.py - Adding a record (no LLM). A record that is a near-copy of one already kept (similarity >=
novelty_duplicate_similarity) is counted, not stored: the bar is high on purpose, because embeddings
barely see negation or tense and a correction looks close to what it corrects. Everything else is kept
in full and split into sentences, each embedded and indexed, so it can be found at once, long before
any sleep. Tool records with an outcome are always kept.
"""

from dataclasses import dataclass

from smruti.config import SmrutiConfig
from smruti.store import Store
from smruti.text import split_sentences


@dataclass
class EncodeResult:
    stored: bool
    episode_id: int | None
    novelty: float
    sentences: int = 0


class Encoder:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def encode(self, space: str, role: str, text: str, outcome: str | None = None, now=None, source: str = "chat",
               speaker: str = "") -> EncodeResult:
        vec = self.store.embed(text)
        nearest = self.store.vec_search("episodes", space, vec, 1)
        novelty = 1.0 - (nearest[0][1] if nearest else 0.0)
        if outcome is None and self.config.enable_novelty_gate and nearest \
                and nearest[0][1] >= self.config.novelty_duplicate_similarity:
            self.store.bump_seen(nearest[0][0])
            return EncodeResult(False, nearest[0][0], novelty)

        episode_id = self.store.add_episode(space, role, text, outcome, novelty, vec, now=now, source=source,
                                            speaker=speaker)
        count = self.split(self.store.episodes([episode_id])[0])
        if nearest:
            # Only kept records that found a neighbour sample the similarity distribution recall calibrates against.
            self.store.push_novelty(space, novelty, self.config.novelty_window)
        return EncodeResult(True, episode_id, novelty, count)

    def split(self, episode) -> int:
        """Stores the sentences of a kept record; returns how many."""
        texts = split_sentences(episode.text, self.config.max_sentence_chars)
        if texts:
            self.store.add_sentences(episode.space, episode, texts, self.store.embed_many(texts))
        return len(texts)
