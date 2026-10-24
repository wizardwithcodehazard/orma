"""rank.py - Score fusion, similarity calibration and use-based strength for recall."""

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

    Each novelty score is 1 - the nearest-neighbour similarity of an incoming record to memory, so
    1 - novelty is the empirical distribution of "how similar the closest memory is" for real traffic.
    The floor is a percentile of it: a similarity at or below it is what an arbitrary record gets by chance
    (embedding noise: small models give unrelated texts ~0.5) and carries no evidence. Above the floor the
    weight rises linearly to 1 at identity. During warm-up or when ablated, every similarity weighs 1."""
    if not enabled or len(novelty_scores) < warmup:
        return lambda sim: 1.0
    floor = float(np.percentile([1.0 - s for s in novelty_scores], percentile))
    if floor >= 1.0:
        return lambda sim: 1.0 if sim >= 1.0 else 0.0
    return lambda sim: min(1.0, max(0.0, (sim - floor) / (1.0 - floor)))


def base_level(times, now: float, decay: float) -> float:
    """ACT-R base-level activation (as strength, not its log): sum over the times a memory was used of
    (age_days + 1) ** -decay. One fresh use gives 1; each older or extra one adds less."""
    return sum((max(0.0, now - t) / 86400.0 + 1.0) ** -decay for t in times)


def hawkes_burst(times, now: float, alpha: float = 0.4, half_life_seconds: float = 1800.0) -> float:
    """Transient self-exciting burst intensity using an exponential Hawkes kernel.
    Recent accesses within the conversational half-life produce an excitation surge
    that decays exponentially back to baseline.
    """
    if not times:
        return 0.0
    beta = 0.69314718056 / max(1.0, half_life_seconds)  # ln(2) / half_life
    return float(sum(alpha * np.exp(-beta * max(0.0, now - t)) for t in times if now >= t))


def ebbinghaus_retention(times, now: float, initial_stability_days: float = 2.0, repetition_gain: float = 0.5) -> float:
    """Ebbinghaus retention score R = exp(-age_days / Stability).
    Stability multiplies with each spaced repetition: S = S0 * (1 + gain * (n - 1)).
    Frequently reinforced facts stay near 1.0, while one-off mentions decay exponentially.
    """
    if not times:
        return 0.0
    n = len(times)
    stability = initial_stability_days * (1.0 + repetition_gain * (n - 1))
    latest = max(times)
    age_days = max(0.0, now - latest) / 86400.0
    return float(np.exp(-age_days / max(0.1, stability)))


def use_weight(strength: float, weight: float) -> float:
    """A bounded boost for strength: (1 - weight) + weight * min(strength, 1), so recency and repetition rank
    among relevant memories but never bury an old relevant one under fresh irrelevant ones."""
    return (1.0 - weight) + weight * min(strength, 1.0)


def spread_activation(source_weights: dict[int, float], adjacency: dict[int, list[int]]) -> dict[int, float]:
    """Spreading activation with the ACT-R Fan Effect.
    Each source node spreads weight / fan to its adjacent target nodes.
    Specific nodes with small fan transmit high activation; broad hubs transmit diluted activation.
    """
    activated: dict[int, float] = {}
    for node, weight in source_weights.items():
        neighbors = adjacency.get(node, [])
        if not neighbors:
            continue
        fan = len(neighbors)
        edge_energy = weight / max(1, fan)
        for neighbor in neighbors:
            activated[neighbor] = activated.get(neighbor, 0.0) + edge_energy
    return activated


