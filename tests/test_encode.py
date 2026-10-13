import pytest

from smruti.encode import Encoder

DISTINCT = ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel", "india juliet"]


def test_first_turns_are_always_stored(store, config):
    enc = Encoder(store, config)
    first = enc.encode("s", "user", "I live in Pune")
    assert first.stored and first.novelty == pytest.approx(1.0)
    again = enc.encode("s", "user", "I live in Pune")
    assert again.stored  # still warming up
    assert again.novelty == pytest.approx(0.0, abs=1e-6)


def test_redundant_turn_is_counted_not_stored_after_warmup(store, config):
    config.novelty_warmup = 4
    enc = Encoder(store, config)
    ids = [enc.encode("s", "user", t).episode_id for t in DISTINCT]
    repeat = enc.encode("s", "user", "alpha bravo")
    assert not repeat.stored and repeat.episode_id == ids[0]
    assert store.episodes([ids[0]])[0].seen_count == 2
    assert len(store.pending_episodes("s", 100)) == 5


def test_tool_outcomes_always_stored(store, config):
    config.novelty_warmup = 4
    enc = Encoder(store, config)
    for t in DISTINCT:
        enc.encode("s", "user", t)
    assert enc.encode("s", "tool", "npm run build", outcome="failure").stored
    assert enc.encode("s", "tool", "npm run build", outcome="failure").stored


def test_gate_can_be_ablated(store, config):
    config.novelty_warmup = 4
    config.enable_novelty_gate = False
    enc = Encoder(store, config)
    for t in DISTINCT:
        enc.encode("s", "user", t)
    assert enc.encode("s", "user", "alpha bravo").stored


def test_fact_duplicate_is_dropped_without_bumping_an_episode(store, config):
    config.novelty_warmup = 4
    enc = Encoder(store, config)
    ids = [enc.encode("s", "user", t).episode_id for t in DISTINCT]
    user = store.user_entity("s")
    pred = store.add_predicate("s", "lives_in", "one")
    place = store.add_entity("s", "Pune", "city")
    store.add_fact("s", user, pred, place, None, 1.0, [ids[0]])
    res = enc.encode("s", "user", "user lives in Pune")
    assert not res.stored and res.episode_id is None
    assert [e.seen_count for e in store.episodes(ids)] == [1] * 5
    assert len(store.pending_episodes("s", 100)) == 5


def test_only_stored_turns_with_a_neighbour_feed_the_novelty_history(store, config):
    config.novelty_warmup = 3
    enc = Encoder(store, config)
    enc.encode("s", "user", "alpha bravo")  # empty space: novelty 1.0 is not a similarity sample
    assert store.novelty_scores("s") == []
    for t in DISTINCT[1:4]:
        enc.encode("s", "user", t)
    assert len(store.novelty_scores("s")) == 3
    dropped = enc.encode("s", "user", "alpha bravo")
    assert not dropped.stored
    assert len(store.novelty_scores("s")) == 3  # near-duplicates the gate drops do not inflate the floor
