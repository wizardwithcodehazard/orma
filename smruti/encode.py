"""
encode.py - Per-turn encoding (no LLM). Like hippocampal gating, a turn that adds nothing
new relative to the space's own recent novelty distribution is counted, not stored.
Tool turns with outcomes are always stored: every outcome is evidence for procedures.
"""

from dataclasses import dataclass

import numpy as np

from smruti.config import SmrutiConfig
from smruti.store import Store

NOVELTY_SOURCES = ("episodes", "facts", "summaries")


@dataclass
class EncodeResult:
    stored: bool
    episode_id: int | None
    novelty: float


class Encoder:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def encode(self, space: str, role: str, text: str, outcome: str | None = None, now=None) -> EncodeResult:
        vec = self.store.embed(text)
        similarities = [sim for kind in NOVELTY_SOURCES for _, sim in self.store.vec_search(kind, space, vec, 1)]
        novelty = 1.0 - max(similarities, default=0.0)
        history = self.store.novelty_scores(space)
        self.store.push_novelty(space, novelty, self.config.novelty_window)

        if outcome is None and self._redundant(novelty, history):
            # Bump only the episode this turn duplicates, never an unrelated one when a
            # fact or summary was the closest match.
            nearest = self.store.vec_search("episodes", space, vec, 1)
            if nearest and nearest[0][1] >= max(similarities) - 1e-6:
                self.store.bump_seen(nearest[0][0])
                return EncodeResult(False, nearest[0][0], novelty)
            return EncodeResult(False, None, novelty)

        episode_id = self.store.add_episode(space, role, text, outcome, novelty, vec, now=now)
        return EncodeResult(True, episode_id, novelty)

    def _redundant(self, novelty: float, history: list[float]) -> bool:
        if not self.config.enable_novelty_gate or len(history) < self.config.novelty_warmup:
            return False
        return novelty < float(np.percentile(history, self.config.novelty_percentile))
