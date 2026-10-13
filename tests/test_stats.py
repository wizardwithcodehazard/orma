from smruti import Memory


def test_stats_count_memory_by_kind(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    s = mem.store
    user = s.user_entity("s")
    lives = s.add_predicate("s", "lives_in", "one")
    old = s.add_fact("s", user, lives, None, "Pune", 1.0, [1])
    s.close_fact(old, 2.0)
    s.add_fact("s", user, lives, None, "Bangalore", 2.0, [2])
    mem.observe("user", "hello", space="s")
    assert mem.stats("s") == {
        "episodes": 1, "pending_episodes": 1, "entities": 1, "current_facts": 1,
        "closed_facts": 1, "procedures": 0, "summaries": 0,
    }
    assert mem.stats("other")["episodes"] == 0
    mem.close()


def test_due_spaces_without_an_llm(config, embedder):
    config.sleep_after_episodes = 1
    mem = Memory(config=config, embedder=embedder)
    mem.observe("user", "hello", space="s")
    assert mem.due_spaces() == ["s"]
    mem.close()
