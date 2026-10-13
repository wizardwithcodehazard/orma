import pytest

from smruti.recall import Recaller
from smruti.recall.pack import FRAMING


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
    assert out.messages[0]["content"].startswith("Be brief.\n\n## Memory\n" + FRAMING)
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
    assert text.startswith("## Memory\n" + FRAMING + "\nKnown facts about the user:\n- user lives in Pune")
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
        "s", [{"role": "user", "content": "I'm baking cookies for my sister, any ideas?"}], budget=120, now=1.0)
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


def warm_up(store, config, space, n=None):
    """Feed the encoder unrelated turns so the space has a real nearest-neighbour similarity distribution."""
    from smruti.encode import Encoder

    encoder = Encoder(store, config)
    for i in range(n or config.novelty_warmup + 10):
        encoder.encode(space, "user", f"w{i}alpha w{i}beta w{i}gamma w{i}delta")


def test_unrelated_query_recalls_nothing_with_a_noisy_embedder(noisy_store, config):
    store = noisy_store
    ravi = store.add_entity("s", "Ravi", "person")
    meera = store.add_entity("s", "Meera", "person")
    store.add_fact("s", ravi, store.add_predicate("s", "plays", "many"), None, "guitar", 1.0, [1], now=1.0)
    store.add_fact("s", meera, store.add_predicate("s", "works_at", "one"), None, "Infosys", 1.0, [1], now=1.0)
    warm_up(store, config, "s")
    assert len(store.novelty_scores("s")) >= config.novelty_warmup

    recaller = Recaller(store, config)
    unrelated = recaller.build("s", [{"role": "user", "content": "what is the weather like today"}], now=1.0)
    assert memory_lines(unrelated.memory_text) == []
    related = recaller.build("s", [{"role": "user", "content": "does Ravi still play guitar"}], now=1.0)
    assert "- Ravi plays guitar" in related.memory_text

    config.enable_similarity_weighting = False  # ablation: plain rank-only RRF lets the junk top hit in
    ablated = Recaller(store, config).build("s", [{"role": "user", "content": "what is the weather like today"}], now=1.0)
    assert memory_lines(ablated.memory_text) != []


def test_entity_name_links_only_when_all_its_terms_are_in_the_query(store, config):
    from smruti.recall.link import link_entities

    ny = store.add_entity("s", "New York", "city")
    no_embedding = lambda sim: 0.0  # isolate lexical linking

    def linked(query):
        return link_entities(store, config, "s", query, store.embed(query), no_embedding)

    assert ny not in linked("any new ideas?")
    assert linked("I visited New York") == {ny: 1.0}
    assert linked("new york is big") == {ny: 1.0}


def test_embedding_links_are_weighted_by_calibrated_similarity(store, config):
    from smruti.recall.link import link_entities

    ny = store.add_entity("s", "New York", "city")
    query = "trip"
    assert link_entities(store, config, "s", query, store.embed(query), lambda sim: 0.0) == {}
    assert link_entities(store, config, "s", query, store.embed(query), lambda sim: 0.3) == {ny: 0.3}


def test_candidate_facts_are_bounded(store, config):
    user = store.user_entity("s")
    note = store.add_predicate("s", "mentioned", "many")
    for i in range(500):
        store.add_fact("s", user, note, None, f"filler {i}", 1.0, [i], now=1.0)
    ranked = Recaller(store, config)._ranked("s", "tell me about user stuff", now=1.0)
    facts = [item for item in ranked if item.section == "facts"]
    assert 0 < len(facts) <= config.max_candidate_facts


def test_pending_episodes_are_recalled_before_sleep(store, config):
    from smruti.encode import Encoder

    Encoder(store, config).encode("s", "user", "my sister Asha is allergic to peanuts", now=1.0)
    result = Recaller(store, config).build("s", [{"role": "user", "content": "what is Asha allergic to"}], now=1.0)
    assert "Recent notes:" in result.memory_text
    assert "- my sister Asha is allergic to peanuts" in result.memory_text

    config.enable_episode_recall = False
    ablated = Recaller(store, config).build("s", [{"role": "user", "content": "what is Asha allergic to"}], now=1.0)
    assert ablated.memory_text == ""


def test_consolidated_episodes_and_current_turns_are_not_recalled_as_notes(store, config):
    from smruti.encode import Encoder

    a = Encoder(store, config).encode("s", "user", "my sister Asha is allergic to peanuts", now=1.0).episode_id
    Encoder(store, config).encode("s", "user", "Asha plays the violin on weekends", now=1.0)
    store.mark_consolidated([a], 2.0)
    query = [{"role": "user", "content": "Asha plays the violin on weekends"}]
    assert Recaller(store, config).build("s", query, now=1.0).memory_text == ""  # b is in the request itself
    other = [{"role": "user", "content": "what does Asha play or eat, peanuts violin"}]
    text = Recaller(store, config).build("s", other, now=1.0).memory_text
    assert "violin on weekends" in text and "allergic" not in text


def test_notes_pack_after_summaries(store, config):
    from smruti.encode import Encoder

    store.upsert_summary("s", "asha", "Asha is a sister who is allergic", 8, now=1.0)
    Encoder(store, config).encode("s", "user", "Asha is allergic to peanuts", now=1.0)
    text = Recaller(store, config).build("s", [{"role": "user", "content": "Asha allergic"}], now=1.0).memory_text
    assert text.index("Background:") < text.index("Recent notes:")


def test_notes_skip_assistant_episodes_and_cap_length(store, config):
    from smruti.encode import Encoder

    config.recall_note_chars = 40
    Encoder(store, config).encode("s", "assistant", "Asha peanuts allergy: here is a long reply about it", now=1.0)
    Encoder(store, config).encode("s", "user", "Asha is allergic to peanuts and also to many other things", now=1.0)
    text = Recaller(store, config).build("s", [{"role": "user", "content": "Asha allergic peanuts"}], now=1.0).memory_text
    notes = memory_lines(text)
    assert len(notes) == 1 and "long reply" not in text
    assert len(notes[0]) - 2 == 40 and notes[0].endswith("…") and notes[0].startswith("- Asha is allergic")


def test_tool_episodes_are_recalled_as_notes(store, config):
    from smruti.encode import Encoder

    Encoder(store, config).encode("s", "tool", "pytest passed for Asha module", now=1.0)
    text = Recaller(store, config).build("s", [{"role": "user", "content": "Asha pytest module"}], now=1.0).memory_text
    assert "- pytest passed for Asha module" in text
