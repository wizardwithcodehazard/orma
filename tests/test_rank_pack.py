import pytest

from smruti.recall.pack import MemoryItem, count_tokens, pack
from smruti.recall.rank import recency, rrf


def test_rrf_rewards_agreement_and_normalizes():
    scores = rrf([1, 2, 3], [3, 1])
    assert scores[1] == 1.0
    assert scores[1] > scores[3] > scores[2]


def test_rrf_of_nothing_is_empty():
    assert rrf([], []) == {}


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
