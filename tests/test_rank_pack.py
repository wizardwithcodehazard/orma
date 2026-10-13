import pytest

from smruti.recall.pack import MemoryItem, count_tokens, pack
from smruti.recall.rank import recency, rrf


def ranked(*ids, weight=1.0):
    return [(i, weight) for i in ids]


def test_rrf_rewards_agreement_and_normalizes():
    s = rrf(ranked(1, 2, 3), ranked(3, 1))
    assert s[1] > s[3] > s[2]
    assert all(0 < v <= 1 for v in s.values())
    assert rrf(ranked(1, 2), ranked(1, 3))[1] == pytest.approx(1.0)
    assert rrf(ranked(1, 2), ranked(3))[1] < 1.0  # first in only one list is not inflated to 1.0


def test_rrf_of_nothing_is_empty():
    assert rrf([], []) == {}


def test_rrf_zero_weight_drops_an_item():
    s = rrf([(1, 0.0), (2, 1.0)], [(1, 0.0)])
    assert 1 not in s
    assert s[2] == pytest.approx((1 / 62) / (2 / 61))  # 2 keeps its rank (second), weight does not shift ranks


def test_rrf_weights_scale_contributions():
    full, half = rrf([(1, 1.0)], []), rrf([(1, 0.5)], [])
    assert half[1] == pytest.approx(full[1] / 2)
    assert rrf([(1, 1.0)], [(1, 1.0)], k=10)[1] == pytest.approx(1.0)  # normalized by len(rankings) / (k + 1)
    assert rrf([(1, 1.0)], [(1, 0.25)], k=10)[1] == pytest.approx((1 + 0.25) / 2)


def test_similarity_weight_is_calibrated_against_the_space_distribution():
    from smruti.recall.rank import similarity_weight

    novelty = [0.6, 0.5, 0.4]  # nearest-neighbour similarities 0.4, 0.5, 0.6 -> median floor 0.5
    w = similarity_weight(novelty, percentile=50.0, warmup=3, enabled=True)
    assert w(0.45) == 0.0 and w(0.5) == 0.0
    assert w(0.75) == pytest.approx(0.5)
    assert w(1.0) == pytest.approx(1.0)
    assert similarity_weight(novelty, 50.0, warmup=4, enabled=True)(0.1) == 1.0  # warming up: plain RRF
    assert similarity_weight(novelty, 50.0, warmup=3, enabled=False)(0.1) == 1.0  # ablated


def test_recency_half_life():
    assert recency(0.0, 14 * 86400.0, 14.0) == pytest.approx(0.5)
    assert recency(100.0, 50.0, 14.0) == 1.0


def test_count_tokens_rounds_up():
    assert count_tokens("abcdefg", 3.5) == 2


def test_pack_keeps_last_turns_and_respects_budget():
    turns = [{"role": "user", "content": "x" * 70}, {"role": "assistant", "content": "y" * 70},
             {"role": "user", "content": "z" * 70}]
    items = [MemoryItem("facts", i, f"fact number {i}") for i in range(50)]
    packed = pack([], items, turns, budget=100, chars_per_token=3.5, min_recent=2)
    assert packed.turns == turns[-2:]
    assert packed.tokens <= 100
    assert 0 < len(packed.used) < 50
    assert packed.memory_text.startswith("## Memory\nRelevant facts:\n- fact number 0")


def test_pack_orders_sections_and_adds_older_turns_when_room():
    core = [MemoryItem("core", 1, "user lives in Pune")]
    ranked = [MemoryItem("summaries", 9, "travel: planning a Goa trip"),
              MemoryItem("procedures", 4, "When build fails: npm install (works)")]
    turns = [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}]
    packed = pack(core, ranked, turns, budget=1000, chars_per_token=3.5, min_recent=1)
    assert packed.turns == turns
    text = packed.memory_text
    assert text.index("Known facts about the user") < text.index("Lessons from past actions") < text.index("Background")


def test_pack_without_memory_has_no_header():
    packed = pack([], [], [{"role": "user", "content": "hi"}], 100, 3.5, 2)
    assert packed.memory_text == "" and packed.used == []
