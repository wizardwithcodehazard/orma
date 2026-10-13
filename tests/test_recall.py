import pytest

from smruti.recall import Recaller


def build_graph(store):
    user = store.user_entity("s")
    pune = store.add_entity("s", "Pune", "city")
    mom = store.add_entity("s", "mother", "person")
    lives = store.add_predicate("s", "lives_in", "one")
    has = store.add_predicate("s", "has_mother", "one")
    return {  # same timestamp so recency cannot reorder them
        "lives": store.add_fact("s", user, lives, pune, None, 1.0, [1], now=1.0),
        "mom": store.add_fact("s", user, has, mom, None, 1.0, [1], now=1.0),
        "mom_city": store.add_fact("s", mom, lives, None, "Nagpur", 1.0, [1], now=1.0),
    }


def memory_lines(text):
    return [line for line in text.splitlines() if line.startswith("- ")]


def test_memory_is_appended_to_the_system_prompt(store, config):
    build_graph(store)
    out = Recaller(store, config).build(
        "s", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Where should I travel?"}]
    )
    assert out.messages[0]["role"] == "system"
    assert out.messages[0]["content"].startswith("Be brief.\n\n## Memory")
    assert "- user lives in Pune" in out.memory_text
    assert out.messages[-1] == {"role": "user", "content": "Where should I travel?"}


def test_graph_expansion_reaches_facts_search_misses(store, config):
    build_graph(store)
    config.search_k = 1
    with_graph = Recaller(store, config).build("s", [{"role": "user", "content": "Any gift ideas?"}])
    assert "- mother lives in Nagpur" in with_graph.memory_text
    assert len(memory_lines(with_graph.memory_text)) == 3

    config.enable_graph = False
    without = Recaller(store, config).build("s", [{"role": "user", "content": "Any gift ideas?"}])
    assert len(memory_lines(without.memory_text)) <= 1


def test_closer_facts_rank_first(store, config):
    build_graph(store)
    text = Recaller(store, config).build("s", [{"role": "user", "content": "hello"}]).memory_text
    assert text.index("user lives in Pune") < text.index("mother lives in Nagpur")


def test_core_block_holds_confident_user_facts(store, config):
    ids = build_graph(store)
    store.reinforce_fact(ids["lives"], [2])
    store.reinforce_fact(ids["lives"], [3])  # support 3 -> confidence 0.75
    text = Recaller(store, config).build("s", [{"role": "user", "content": "hello"}]).memory_text
    assert text.startswith("## Memory\nKnown facts about the user:\n- user lives in Pune")
    assert text.count("user lives in Pune") == 1


def test_budget_is_respected(store, config):
    user = store.user_entity("s")
    likes = store.add_predicate("s", "likes", "many")
    for i in range(200):
        store.add_fact("s", user, likes, None, f"hobby number {i} with a long description", 1.0, [i])
    out = Recaller(store, config).build("s", [{"role": "user", "content": "what do I like?"}], budget=300)
    assert out.tokens <= 300
    assert 0 < len(out.used) < 200


def test_recall_does_not_write(store, config):
    build_graph(store)
    before = [(f.id, f.last_used) for f in store.current_facts("s")]
    Recaller(store, config).build("s", [{"role": "user", "content": "Where do I live?"}], now=9e9)
    assert [(f.id, f.last_used) for f in store.current_facts("s")] == before


def test_recall_on_empty_space_creates_nothing(store, config):
    out = Recaller(store, config).build("s", [{"role": "user", "content": "hello"}])
    assert out.memory_text == ""
    assert store.find_entity("s", "user") is None


def test_exact_token_accounting_with_system_prompt(store, config):
    import math

    user = store.user_entity("s")
    likes = store.add_predicate("s", "likes", "many")
    for i in range(100):
        store.add_fact("s", user, likes, None, f"hobby number {i} with a long description", 1.0, [i])
    turns = [{"role": "user", "content": "what do I like?"}]
    cpt = config.chars_per_token
    for budget in range(150, 400, 7):
        out = Recaller(store, config).build("s", [{"role": "system", "content": "Be brief."}, *turns], budget=budget)
        assert out.tokens <= budget
        expected = math.ceil(len(out.messages[0]["content"]) / cpt) + math.ceil(len(turns[0]["content"]) / cpt)
        assert out.tokens == expected


def test_no_user_message_skips_search(store, config):
    build_graph(store)
    out = Recaller(store, config).build("s", [{"role": "system", "content": "Be brief."}])
    assert memory_lines(out.memory_text) == []


def test_hub_fan_does_not_drown_cue_related_facts(store, config):
    user = store.user_entity("s")
    asha = store.add_entity("s", "Asha", "person")
    mentioned = store.add_predicate("s", "mentioned", "many")
    for i in range(60):
        store.add_fact("s", user, mentioned, None, f"topic {i} about project {i % 7}", 1.0, [i], now=1.0)
    store.add_fact("s", user, store.add_predicate("s", "has_sister", "many"), asha, None, 1.0, [100], now=1.0)
    store.add_fact("s", asha, store.add_predicate("s", "is_allergic_to", "many"), None, "peanuts", 1.0, [101], now=1.0)
    out = Recaller(store, config).build(
        "s", [{"role": "user", "content": "I'm baking cookies for my sister, any ideas?"}], budget=80, now=1.0)
    assert 3 <= len(memory_lines(out.memory_text)) <= 6
    assert "- Asha is allergic to peanuts" in out.memory_text


def test_fan_divides_activation_among_an_entitys_facts(store):
    from smruti.recall.activation import spread

    user = store.user_entity("s")
    asha = store.add_entity("s", "Asha", "person")
    note = store.add_predicate("s", "mentioned", "many")
    fillers = [store.add_fact("s", user, note, None, f"topic {i}", 1.0, [1]) for i in range(9)]
    sister = store.add_fact("s", user, store.add_predicate("s", "has_sister", "many"), asha, None, 1.0, [1])
    allergy = store.add_fact("s", asha, store.add_predicate("s", "is_allergic_to", "many"), None, "peanuts", 1.0, [1])

    act = spread(store, "s", {user: 1.0}, max_hops=2, decay=0.5)
    assert act[fillers[0]] == pytest.approx(0.1)  # user's 1.0 split over 10 facts
    assert act[allergy] == pytest.approx(0.1 * 0.5 / 2)  # Asha got 0.1, decayed, split over 2 facts
    assert act[sister] == pytest.approx(0.1 + 0.025)  # from user at step 0 and from Asha at step 1

    act = spread(store, "s", {asha: 1.0}, max_hops=1, decay=0.5)
    assert act == {sister: pytest.approx(0.5), allergy: pytest.approx(0.5)}
