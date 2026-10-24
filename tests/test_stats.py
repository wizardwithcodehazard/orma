from smruti import Memory


def test_stats_count_memory_by_kind(config, embedder, seed_state):
    mem = Memory(config=config, embedder=embedder)
    s = mem.store
    old = seed_state(s, "s", "home_city", "Pune", now=1.0)
    s.close_state(old, 2.0)
    seed_state(s, "s", "home_city", "Bangalore", now=2.0)
    mem.observe("user", "hello there. how are you", space="s")
    assert mem.stats("s") == {"records": 3, "pending_records": 1, "sentences": 4, "entities": 1,
                              "current_states": 1, "past_states": 1, "events": 0}
    assert mem.stats("other")["records"] == 0
    mem.close()


def test_due_spaces_without_an_llm(config, embedder):
    config.sleep_after_episodes = 1
    mem = Memory(config=config, embedder=embedder)
    mem.observe("user", "hello", space="s")
    assert mem.due_spaces() == ["s"]
    mem.close()
