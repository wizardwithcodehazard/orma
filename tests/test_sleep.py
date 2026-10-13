from smruti.encode import Encoder
from smruti.llm import LLMError
from smruti.sleep import Sleeper

PUNE = {"entities": [{"name": "Pune", "type": "city"}],
        "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]}
MOTHER = {"entities": [{"name": "mother", "type": "person"}],
          "facts": [{"subject": "user", "predicate": "has_mother", "object": "mother", "cardinality": "one"},
                    {"subject": "mother", "predicate": "lives_in", "object": "Nagpur", "cardinality": "one"}]}


def test_sleep_consolidates_in_batches(store, config, fake_llm):
    config.sleep_batch_episodes = 2
    enc = Encoder(store, config)
    for text in ["I live in Pune", "nice weather", "my mom lives in Nagpur"]:
        enc.encode("s", "user", text)
    llm = fake_llm([PUNE, MOTHER])
    report = Sleeper(store, config, llm).run("s")
    assert (report.batches, report.episodes, report.failed_batches) == (2, 3, 0)
    assert report.consolidation.facts_added == 3
    assert store.pending_episodes("s", 10) == []
    assert "KNOWN PREDICATES: lives_in" in llm.calls[1][1]["content"]


def test_failed_batch_is_retried_then_retired(store, config, fake_llm):
    config.sleep_max_attempts = 2
    Encoder(store, config).encode("s", "user", "hello")
    sleeper = Sleeper(store, config, fake_llm([LLMError("down"), LLMError("down")]))
    assert sleeper.run("s").failed_batches == 1
    assert len(store.pending_episodes("s", 10)) == 1
    assert sleeper.run("s").failed_batches == 1
    assert store.pending_episodes("s", 10) == []


def test_due_spaces_by_count_or_idle_time(store, config, fake_llm):
    config.sleep_after_episodes = 2
    config.sleep_idle_seconds = 60
    enc = Encoder(store, config)
    enc.encode("busy", "user", "one", now=1000.0)
    enc.encode("busy", "user", "two", now=1000.0)
    enc.encode("quiet", "user", "three", now=1000.0)
    enc.encode("fresh", "user", "four", now=1050.0)
    assert Sleeper(store, config, fake_llm()).due_spaces(now=1070.0) == ["busy", "quiet"]


def test_consolidated_episodes_are_pruned_after_retention(store, config, fake_llm):
    config.episode_retention_days = 1
    Encoder(store, config).encode("s", "user", "hello", now=0.0)
    Sleeper(store, config, fake_llm([{}])).run("s", now=10.0)
    report = Sleeper(store, config, fake_llm()).run("s", now=10.0 + 2 * 86400)
    assert report.pruned_episodes == 1
