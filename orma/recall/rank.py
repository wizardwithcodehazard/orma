"""rank.py - Score fusion, similarity calibration and use-based strength for recall."""

from collections.abc import Callable
from typing import Any

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


def similarity_weight(
    novelty_scores: list[float], percentile: float, warmup: int, enabled: bool
) -> Callable[[float], float]:
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


def ebbinghaus_retention(
    times, now: float, initial_stability_days: float = 2.0, repetition_gain: float = 0.5
) -> float:
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


def personalized_pagerank(
    seeds: dict[Any, float],
    adjacency: dict[Any, list[Any]],
    alpha: float = 0.85,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> dict[Any, float]:
    """Personalized PageRank (PPR) via Power Iteration on Memory Association Graph.
    Propagates activation from query seed nodes across multi-hop entity/state graph:
    p^{(t+1)} = (1 - alpha) * p_0 + alpha * W * p^{(t)}
    where W is the row/column normalized random-walk transition matrix.
    """
    if not seeds:
        return {}

    seed_sum = sum(seeds.values())
    p0 = {node: w / seed_sum for node, w in seeds.items()} if seed_sum > 1e-9 else dict(seeds)

    all_nodes = set(p0.keys())
    for neighbors in adjacency.values():
        all_nodes.update(neighbors)

    p = dict(p0)

    for _ in range(max_iter):
        next_p: dict[int, float] = {}

        # Teleportation component: (1 - alpha) * p_0
        for node, val in p0.items():
            next_p[node] = (1.0 - alpha) * val

        # Random walk propagation: alpha * W * p
        for node, score in p.items():
            if score <= 0.0:
                continue
            neighbors = adjacency.get(node, [])
            if neighbors:
                deg = len(neighbors)
                share = (alpha * score) / deg
                for nbr in neighbors:
                    next_p[nbr] = next_p.get(nbr, 0.0) + share
            else:
                # Dangling node teleports back to p_0
                for s_node, s_val in p0.items():
                    next_p[s_node] = next_p.get(s_node, 0.0) + (alpha * score) * s_val

        diff = sum(abs(next_p.get(n, 0.0) - p.get(n, 0.0)) for n in all_nodes)
        p = next_p
        if diff < tol:
            break

    return p


def mmr(
    candidates: list[tuple[int, float]],
    vectors: dict[int, np.ndarray],
    k: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
    Iteratively selects candidates balancing relevance against redundancy with already selected items.
    MMR = argmax_{i in Remaining} [ lambda * score_i - (1 - lambda) * max_{j in Selected} CosineSim(v_i, v_j) ]
    """
    if not candidates or k <= 0:
        return []

    max_score = max(score for _, score in candidates)
    norm_scores = {
        item_id: (score / max_score if max_score > 1e-9 else 1.0) for item_id, score in candidates
    }

    remaining = [item_id for item_id, _ in candidates]
    selected: list[int] = []

    # Normalize vectors to unit sphere for fast cosine similarity via dot product
    unit_vectors: dict[int, np.ndarray] = {}
    for item_id in remaining:
        v = vectors.get(item_id)
        if v is not None:
            norm = float(np.linalg.norm(v))
            unit_vectors[item_id] = v / norm if norm > 1e-9 else v

    while remaining and len(selected) < k:
        if not selected:
            best_id = remaining[0]
            selected.append(best_id)
            remaining.remove(best_id)
            continue

        best_id = remaining[0]
        best_mmr = -1e9

        for item_id in remaining:
            rel = norm_scores.get(item_id, 0.0)
            v_i = unit_vectors.get(item_id)
            if v_i is not None and any(s in unit_vectors for s in selected):
                sim_to_selected = max(
                    float(np.dot(v_i, unit_vectors[s])) for s in selected if s in unit_vectors
                )
            else:
                sim_to_selected = 0.0

            mmr_score = lambda_param * rel - (1.0 - lambda_param) * sim_to_selected
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_id = item_id

        selected.append(best_id)
        remaining.remove(best_id)

    return selected


def rocchio_expand(
    query_vec: np.ndarray,
    feedback_vectors: list[np.ndarray],
    alpha: float = 0.7,
    beta: float = 0.3,
) -> np.ndarray:
    """Rocchio Pseudo-Relevance Feedback (Vector Space Query Expansion).
    Shifts the initial query vector towards the centroid of top pseudo-relevant documents:
    v_expanded = alpha * v_query + (beta / |R|) * sum_{d in R} v_d
    """
    if not feedback_vectors:
        return query_vec

    centroid = np.mean(feedback_vectors, axis=0)
    expanded = alpha * np.asarray(query_vec, dtype=np.float32) + beta * np.asarray(
        centroid, dtype=np.float32
    )
    norm = float(np.linalg.norm(expanded))
    return expanded / norm if norm > 1e-9 else expanded
