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
    Sleeper(store, config, fake_llm([{"facts": []}])).run("s", now=10.0)
    report = Sleeper(store, config, fake_llm()).run("s", now=10.0 + 2 * 86400)
    assert report.pruned_episodes == 1


def test_unexpected_extract_or_apply_error_counts_as_failed_attempt(store, config, fake_llm):
    config.sleep_max_attempts = 1
    Encoder(store, config).encode("s", "user", "hello")
    assert Sleeper(store, config, fake_llm([ValueError("bad")])).run("s").failed_batches == 1
    assert store.pending_episodes("s", 10) == []

    Encoder(store, config).encode("s", "user", "again")
    sleeper = Sleeper(store, config, fake_llm([PUNE]))

    def boom(*args, **kwargs):
        raise RuntimeError("embedder died")

    sleeper.consolidator.apply = boom
    assert sleeper.run("s").failed_batches == 1
    assert store.pending_episodes("s", 10) == []


def test_second_sleeper_does_nothing_while_the_lease_is_held(store, config, fake_llm):
    Encoder(store, config).encode("s", "user", "I live in Pune")
    assert store.acquire_sleep_lease("s", "someone-else", ttl=600, now=__import__("time").time())
    llm = fake_llm([PUNE])
    report = Sleeper(store, config, llm).run("s")
    assert report.batches == 0 and llm.calls == []
    assert len(store.pending_episodes("s", 10)) == 1
    store.release_sleep_lease("s", "someone-else")
    assert Sleeper(store, config, llm).run("s").batches == 1
    assert store.acquire_sleep_lease("s", "next", ttl=600, now=__import__("time").time())  # released afterwards


def test_batches_respect_the_prompt_token_budget(store, config, fake_llm):
    config.sleep_prompt_tokens = 30
    config.sleep_max_episode_chars = 1000
    enc = Encoder(store, config)
    for text in ["aa " * 20, "bb " * 20, "cc " * 20]:  # ~60 chars = ~18 tokens each
        enc.encode("s", "user", text)
    llm = fake_llm([{"facts": []}] * 3)
    report = Sleeper(store, config, llm).run("s")
    assert (report.batches, report.episodes) == (3, 3)  # one per batch, none over budget
    for call in llm.calls:
        log = call[1]["content"].split("<log>\n")[1].split("\n</log>")[0]
        assert len(log) <= config.sleep_prompt_tokens * config.chars_per_token


def test_oversized_episode_is_truncated_to_the_budget(store, config, fake_llm):
    config.sleep_prompt_tokens = 20
    Encoder(store, config).encode("s", "tool", "x" * 5000)
    llm = fake_llm([{"facts": []}])
    assert Sleeper(store, config, llm).run("s").episodes == 1
    log = llm.calls[0][1]["content"].split("<log>\n")[1].split("\n</log>")[0]
    assert 0 < len(log) <= 20 * config.chars_per_token


def test_episode_char_cap_is_configurable(store, config, fake_llm):
    config.sleep_max_episode_chars = 10
    Encoder(store, config).encode("s", "user", "y" * 100)
    llm = fake_llm([{"facts": []}])
    Sleeper(store, config, llm).run("s")
    content = llm.calls[0][1]["content"]
    assert "y" * 10 in content and "y" * 11 not in content


def test_context_lists_only_the_nearest_predicates(store, config, fake_llm):
    config.sleep_context_predicates = 1
    for name in ["lives_in", "works_at", "likes"]:
        store.add_predicate("s", name, "one")
    Encoder(store, config).encode("s", "user", "where does the user work, works at")
    llm = fake_llm([{"facts": []}])
    Sleeper(store, config, llm).run("s")
    assert "KNOWN PREDICATES: works_at\n" in llm.calls[0][1]["content"]


def test_known_entities_show_aliases(store, config, fake_llm):
    store.add_alias("s", "Sahil", store.user_entity("s"))
    Encoder(store, config).encode("s", "user", "hello there")
    llm = fake_llm([{"facts": []}])
    Sleeper(store, config, llm).run("s")
    assert "user (also called: sahil)" in llm.calls[0][1]["content"]
