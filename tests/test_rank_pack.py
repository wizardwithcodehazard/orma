import pytest

from orma.recall.pack import FRAMING, MemoryItem, pack, render
from orma.recall.rank import base_level, rrf, use_weight
from orma.text import count_tokens


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
    from orma.recall.rank import similarity_weight

    novelty = [0.6, 0.5, 0.4]  # nearest-neighbour similarities 0.4, 0.5, 0.6 -> median floor 0.5
    w = similarity_weight(novelty, percentile=50.0, warmup=3, enabled=True)
    assert w(0.45) == 0.0 and w(0.5) == 0.0
    assert w(0.75) == pytest.approx(0.5)
    assert w(1.0) == pytest.approx(1.0)
    assert similarity_weight(novelty, 50.0, warmup=4, enabled=True)(0.1) == 1.0  # warming up: plain RRF
    assert similarity_weight(novelty, 50.0, warmup=3, enabled=False)(0.1) == 1.0  # ablated


def test_base_level_activation_sums_power_law_decayed_evidence():
    day, now = 86400.0, 100 * 86400.0
    assert base_level([now], now, 0.5) == pytest.approx(1.0)  # one fresh evidence
    assert base_level([now - 3 * day], now, 0.5) == pytest.approx(0.5)  # (3 + 1) ** -0.5: older is weaker
    assert base_level([now, now - 3 * day], now, 0.5) == pytest.approx(1.5)  # more evidence is stronger
    assert base_level([now + day], now, 0.5) == pytest.approx(1.0)  # a future stamp counts as fresh
    assert base_level([], now, 0.5) == 0.0


def test_use_weight_is_a_bounded_boost():
    assert use_weight(1.0, 0.3) == pytest.approx(1.0)
    assert use_weight(0.0, 0.3) == pytest.approx(0.7)  # an old memory keeps most of its rank
    assert use_weight(5.0, 0.3) == pytest.approx(1.0)  # strength beyond one adds nothing


def test_hawkes_burst_temporal_excitation():
    from orma.recall.rank import hawkes_burst

    now = 10000.0
    # Fresh event gets full alpha amplitude
    assert hawkes_burst([now], now, alpha=0.4, half_life_seconds=1800.0) == pytest.approx(0.4)
    # Event at exactly half-life (1800s ago) gets alpha * 0.5
    assert hawkes_burst([now - 1800.0], now, alpha=0.4, half_life_seconds=1800.0) == pytest.approx(0.2)
    # Event 2 half-lives ago (3600s ago) gets alpha * 0.25
    assert hawkes_burst([now - 3600.0], now, alpha=0.4, half_life_seconds=1800.0) == pytest.approx(0.1)
    # Event far in the past (1 day ago) decays to near-zero
    assert hawkes_burst([now - 86400.0], now, alpha=0.4, half_life_seconds=1800.0) < 1e-6
    # Empty list yields 0.0
    assert hawkes_burst([], now) == 0.0


def test_ebbinghaus_retention_stability_scaling():
    from orma.recall.rank import ebbinghaus_retention

    day = 86400.0
    now = 100 * day
    # Single mention (n=1, stability = 2.0 days): at t=0, retention is 1.0
    assert ebbinghaus_retention([now], now, initial_stability_days=2.0) == pytest.approx(1.0)
    # After 2 days (t = 1 stability interval): retention is exp(-1) ~ 0.3678
    assert ebbinghaus_retention([now - 2 * day], now, initial_stability_days=2.0) == pytest.approx(0.367879, rel=1e-3)
    # Repeated 5 times (n=5, stability = 2.0 * (1 + 0.5 * 4) = 6.0 days):
    # After 2 days, retention is exp(-2/6) = exp(-0.3333) ~ 0.7165 (much stronger retention!)
    times_5 = [now - (5 - i) * day for i in range(5)]
    times_5[-1] = now - 2 * day  # latest was 2 days ago
    assert ebbinghaus_retention([], now) == 0.0


def test_spread_activation_fan_effect():
    from orma.recall.rank import spread_activation

    sources = {1: 1.0, 2: 1.0}
    adjacency = {
        1: [101, 102],
        2: [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
    }
    act = spread_activation(sources, adjacency)
    # Specific node (fan=2) transmits 1.0 / 2 = 0.5
    assert act[101] == pytest.approx(0.5)
    assert act[102] == pytest.approx(0.5)
    # Generic hub (fan=10) transmits 1.0 / 10 = 0.1
    assert act[201] == pytest.approx(0.1)
    # Specific target receives 5x more focused activation
    assert act[101] == pytest.approx(5 * act[201])
    assert spread_activation({}, adjacency) == {}
    assert spread_activation(sources, {}) == {}





def item(section, i, text, order=0.0):
    return MemoryItem(section, i, text, order)


def test_render_groups_sections_and_orders_things_said_by_time():
    text = render([item("passages", 2, "later", 20.0), item("events", 1, "2026-10-10: trek"),
                   item("states", 3, "user · home city: Pune"), item("passages", 1, "earlier", 10.0)], "2026-10-15")
    assert text.splitlines() == ["## Memory", f"{FRAMING} Today is 2026-10-15.", "Now true:",
                                 "- user · home city: Pune", "Things said before:", "- earlier", "- later",
                                 "Events:", "- 2026-10-10: trek"]
    assert render([]) == ""


def test_pack_keeps_the_last_turns_and_respects_the_budget():
    turns = [{"role": "user", "content": "x" * 70}, {"role": "assistant", "content": "y" * 70},
             {"role": "user", "content": "z" * 35}]
    items = [item("passages", i, f"passage number {i} " * 3, i) for i in range(30)]
    packed = pack(items, turns, budget=150, chars_per_token=3.5, min_recent=1)
    assert packed.turns[-1] == turns[-1] and packed.tokens <= 150 and 0 < len(packed.used) < 30
    assert packed.tokens == count_tokens(packed.memory_text, 3.5) + sum(
        count_tokens(t["content"], 3.5) for t in packed.turns)


def test_pack_fills_leftover_budget_with_older_turns():
    turns = [{"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"},
             {"role": "user", "content": "new question"}]
    packed = pack([item("states", 1, "user · home city: Pune")], turns, budget=500, chars_per_token=3.5,
                  min_recent=1)
    assert packed.turns == turns and "Now true:" in packed.memory_text


def test_states_beyond_their_cap_are_left_out():
    states = [item("states", i, f"user · attribute {i}: some value") for i in range(10)]
    packed = pack(states + [item("passages", 99, "a thing said")], [], budget=1000, chars_per_token=3.5,
                  min_recent=0, state_cap=30)
    used = [i.section for i in packed.used]
    assert used.count("states") < 10 and "passages" in used


def test_pack_without_memory_has_no_header():
    assert pack([], [{"role": "user", "content": "hi"}], 100, 3.5, 2).memory_text == ""
