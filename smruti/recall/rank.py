"""rank.py - Score fusion and time decay for recall."""


def rrf(*rankings: list[int], k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion of id rankings, normalized so the best item scores 1.0."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    top = max(scores.values(), default=0.0)
    return {item: score / top for item, score in scores.items()} if top else {}


def recency(last_used: float, now: float, half_life_days: float) -> float:
    """Exponential decay with the given half-life; never above 1."""
    age_days = max(0.0, now - last_used) / 86400.0
    return 0.5 ** (age_days / half_life_days)
