import pytest

from smruti.encode import Encoder

DISTINCT = ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel", "india juliet"]


def test_first_record_is_kept_and_an_exact_repeat_is_counted(store, config):
    enc = Encoder(store, config)
    first = enc.encode("s", "user", "I live in Pune")
    assert first.stored and first.novelty == pytest.approx(1.0) and first.sentences == 1
    again = enc.encode("s", "user", "I live in Pune")
    assert not again.stored and again.episode_id == first.episode_id
    assert store.episodes([first.episode_id])[0].seen_count == 2


def test_similar_but_different_record_is_kept(store, config):
    """A correction shares most words with what it corrects; only near-copies are dropped."""
    enc = Encoder(store, config)
    enc.encode("s", "user", "I am allergic to peanuts")
    assert enc.encode("s", "user", "I am not allergic to peanuts").stored


def test_records_are_split_into_indexed_sentences(store, config):
    result = Encoder(store, config).encode("s", "document", "Leave goes to Priya. The office closes at 7 pm.",
                                           now=50.0, source="file:handbook.md", speaker="handbook.md")
    sentences = store.episode_sentences([result.episode_id])
    assert [(s.text, s.role, s.speaker, s.created_at) for s in sentences] == [
        ("Leave goes to Priya.", "document", "handbook.md", 50.0),
        ("The office closes at 7 pm.", "document", "handbook.md", 50.0)]
    assert store.episodes([result.episode_id])[0].source == "file:handbook.md"
    assert [hit.id for hit in store.text_search("sentences", "s", "office", 5)] == [sentences[1].id]


def test_tool_outcomes_always_stored(store, config):
    enc = Encoder(store, config)
    assert enc.encode("s", "tool", "npm run build", outcome="failure").stored
    assert enc.encode("s", "tool", "npm run build", outcome="failure").stored


def test_gate_can_be_ablated(store, config):
    config.enable_novelty_gate = False
    enc = Encoder(store, config)
    enc.encode("s", "user", "alpha bravo")
    assert enc.encode("s", "user", "alpha bravo").stored


def test_only_kept_records_with_a_neighbour_feed_the_novelty_history(store, config):
    enc = Encoder(store, config)
    enc.encode("s", "user", "alpha bravo")  # empty space: novelty 1.0 is not a similarity sample
    assert store.novelty_scores("s") == []
    for t in DISTINCT[1:4]:
        enc.encode("s", "user", t)
    assert len(store.novelty_scores("s")) == 3
    assert not enc.encode("s", "user", "alpha bravo").stored
    assert len(store.novelty_scores("s")) == 3  # dropped near-copies do not inflate the floor
