"""rank.py - Score fusion, similarity calibration and time decay for recall."""

from collections.abc import Callable

import numpy as np


def rrf(*rankings: list[tuple[int, float]], k: int = 60) -> dict[int, float]:
    """Weighted reciprocal-rank fusion of (id, weight) rankings: an item at rank r contributes
    weight / (k + r + 1). Normalized by the best possible score (first in every ranking at weight 1),
    so an unrelated top hit is not inflated to 1.0; items whose total is 0 are dropped."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (item, weight) in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    top = len(rankings) / (k + 1)
    return {item: score / top for item, score in scores.items() if score > 0.0}


def similarity_weight(novelty_scores: list[float], percentile: float, warmup: int,
                      enabled: bool) -> Callable[[float], float]:
    """Calibrates a raw cosine similarity against the space's own similarity distribution.

    Each novelty score is 1 - the nearest-neighbour similarity of an incoming turn to memory, so
    1 - novelty is the empirical distribution of "how similar the closest memory is" for real traffic.
    The floor is a percentile of it: at the median, half of all turns found a neighbour at least that
    close, so a similarity at or below it is what an arbitrary turn gets by chance (embedding noise:
    small models give unrelated texts ~0.5) and carries no evidence. Above the floor the weight rises
    linearly to 1 at identity. During warm-up or when ablated, every similarity weighs 1 (plain RRF)."""
    if not enabled or len(novelty_scores) < warmup:
        return lambda sim: 1.0
    floor = float(np.percentile([1.0 - s for s in novelty_scores], percentile))
    if floor >= 1.0:
        return lambda sim: 1.0 if sim >= 1.0 else 0.0
    return lambda sim: min(1.0, max(0.0, (sim - floor) / (1.0 - floor)))


def base_level(evidence_times, now: float, decay: float) -> float:
    """ACT-R base-level activation (as strength, not its log): sum over evidence times t of
    (age_days(t) + 1) ** -decay. One fresh evidence gives 1; each older or extra one adds less."""
    return sum((max(0.0, now - t) / 86400.0 + 1.0) ** -decay for t in evidence_times)


def importance_factor(importance: float, weight: float) -> float:
    """Blend toward 1: with weight w, importance 0 scores 1 - w and importance 1 scores 1."""
    return (1.0 - weight) + weight * importance


def recency(last_used: float, now: float, half_life_days: float) -> float:
    """Exponential decay with the given half-life; never above 1."""
    age_days = max(0.0, now - last_used) / 86400.0
    return 0.5 ** (age_days / half_life_days)
