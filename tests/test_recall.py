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
    lines = text.splitlines()
    assert lines[0] == "## Memory" and lines[1].startswith(FRAMING)
    assert lines[2:4] == ["Known facts about the user:", "- user lives in Pune"]
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

    act, events, entities = spread(store, "s", {user: 1.0}, max_hops=2, decay=0.5)
    assert act[fillers[0]] == pytest.approx(0.1)  # user's 1.0 split over 10 facts
    assert act[allergy] == pytest.approx(0.1 * 0.5 / 2)  # Asha got 0.1, decayed, split over 2 facts
    assert act[sister] == pytest.approx(0.1 + 0.025)  # from user at step 0 and from Asha at step 1
    assert entities == {user: pytest.approx(1.0), asha: pytest.approx(0.1)}  # the cue and what spreading reached

    assert events == {}
    act, _, _ = spread(store, "s", {asha: 1.0}, max_hops=1, decay=0.5)
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


MARCH_7 = 1772841600.0  # 2026-03-07 00:00 UTC


def test_an_entitys_activation_is_shared_between_its_facts_and_events(store):
    from smruti.recall.activation import spread

    user = store.user_entity("s")
    asha = store.add_entity("s", "Asha", "person")
    allergy = store.add_fact("s", asha, store.add_predicate("s", "is_allergic_to", "many"), None, "peanuts", 1.0, [1])
    trip = store.add_event("s", "user went to Goa with Asha", MARCH_7, True, [user, asha], [1])

    facts, events, _ = spread(store, "s", {asha: 1.0}, max_hops=1, decay=0.5)
    assert facts == {allergy: pytest.approx(0.5)} and events == {trip: pytest.approx(0.5)}
    facts, events, _ = spread(store, "s", {asha: 1.0}, max_hops=1, decay=0.5, events=False)
    assert facts == {allergy: pytest.approx(1.0)} and events == {}


def test_events_are_recalled_with_their_date_through_a_participant(noisy_store, config):
    store = noisy_store
    user = store.user_entity("s")
    asha = store.add_entity("s", "Asha", "person")
    store.add_event("s", "user baked a chocolate cake for a birthday party", MARCH_7, True, [user, asha], [1],
                    now=MARCH_7)
    warm_up(store, config, "s")
    query = [{"role": "user", "content": "how is Asha doing"}]

    text = Recaller(store, config).build("s", query, now=MARCH_7).memory_text
    assert "Past events:\n- 2026-03-07: user baked a chocolate cake for a birthday party" in text
    config.enable_graph = False  # the gist never mentions Asha: only spreading reaches it
    assert "Past events" not in Recaller(store, config).build("s", query, now=MARCH_7).memory_text


def test_events_are_found_by_gist_search(store, config):
    user = store.user_entity("s")
    store.add_event("s", "user ran a half marathon in Pune", MARCH_7, True, [user], [1], now=MARCH_7)
    query = [{"role": "user", "content": "when did I run the half marathon?"}]
    text = Recaller(store, config).build("s", query, now=MARCH_7).memory_text
    assert "- 2026-03-07: user ran a half marathon in Pune" in text

    config.enable_events = False
    assert "Past events" not in Recaller(store, config).build("s", query, now=MARCH_7).memory_text


def test_events_pack_between_facts_and_summaries(store, config):
    user = store.user_entity("s")
    marathon = store.add_entity("s", "marathon", "event")
    store.add_fact("s", user, store.add_predicate("s", "trains_for", "many"), marathon, None, MARCH_7, [1], now=MARCH_7)
    store.add_event("s", "user ran a marathon", MARCH_7, True, [user, marathon], [1], now=MARCH_7)
    store.upsert_summary("s", "running", "user runs a marathon every year", 8, now=MARCH_7)
    text = Recaller(store, config).build("s", [{"role": "user", "content": "marathon"}], now=MARCH_7).memory_text
    assert text.index("Relevant facts:") < text.index("Past events:") < text.index("Background:")


def test_candidate_events_are_bounded(store, config):
    user = store.user_entity("s")
    for i in range(40):
        store.add_event("s", f"user went running on day {i}", MARCH_7, True, [user], [i], now=MARCH_7)
    config.max_candidate_events = 5
    ranked = Recaller(store, config)._ranked("s", "user running", now=MARCH_7)
    assert len([item for item in ranked if item.section == "events"]) == 5


# --- Priming -----------------------------------------------------------------------------

RAVI = [{"role": "user", "content": "does Ravi still play guitar"}]
WEATHER = [{"role": "user", "content": "what is the weather like today"}]


def people(store, config):
    ravi = store.add_entity("s", "Ravi", "person")
    meera = store.add_entity("s", "Meera", "person")
    store.add_fact("s", ravi, store.add_predicate("s", "plays", "many"), None, "guitar", 1.0, [1], now=1.0)
    store.add_fact("s", meera, store.add_predicate("s", "works_at", "one"), None, "Infosys", 1.0, [1], now=1.0)
    warm_up(store, config, "s")


def test_priming_keeps_recently_activated_entities_warm(noisy_store, config):
    people(noisy_store, config)
    recaller = Recaller(noisy_store, config)
    recaller.build("s", RAVI, now=1.0)
    follow_up = recaller.build("s", WEATHER, now=1.0)  # never mentions Ravi
    assert memory_lines(follow_up.memory_text) == ["- Ravi plays guitar"]
    fresh = Recaller(noisy_store, config).build("s", WEATHER, now=1.0)  # a new process starts cold
    assert memory_lines(fresh.memory_text) == []


def test_priming_fades_over_unrelated_requests(noisy_store, config):
    people(noisy_store, config)
    recaller = Recaller(noisy_store, config)
    recaller.build("s", RAVI, now=1.0)
    strengths = []
    for _ in range(8):
        recaller.build("s", WEATHER, now=1.0)
        strengths.append(max(recaller.primed("s").values(), default=0.0))
    assert strengths == sorted(strengths, reverse=True) and strengths[0] > 0.0
    assert recaller.primed("s") == {}
    assert memory_lines(recaller.build("s", WEATHER, now=1.0).memory_text) == []


def test_spaces_do_not_share_priming(noisy_store, config):
    people(noisy_store, config)
    recaller = Recaller(noisy_store, config)
    recaller.build("s", RAVI, now=1.0)
    primed = recaller.primed("s")
    recaller.build("t", WEATHER, now=1.0)
    assert recaller.primed("t") == {}
    assert recaller.primed("s") == primed  # another space's requests do not decay it


def test_priming_can_be_disabled(noisy_store, config):
    people(noisy_store, config)
    config.enable_priming = False
    recaller = Recaller(noisy_store, config)
    recaller.build("s", RAVI, now=1.0)
    assert recaller.primed("s") == {}
    assert memory_lines(recaller.build("s", WEATHER, now=1.0).memory_text) == []


def test_priming_keeps_only_the_strongest_entities(store, config):
    user = store.user_entity("s")
    likes = store.add_predicate("s", "knows", "many")
    for i in range(30):
        store.add_fact("s", user, likes, store.add_entity("s", f"friend{i}", "person"), None, 1.0, [i], now=1.0)
    config.priming_size = 4
    recaller = Recaller(store, config)
    recaller.build("s", [{"role": "user", "content": "who do I know"}], now=1.0)
    assert 0 < len(recaller.primed("s")) <= 4


def test_concurrent_builds_share_priming_safely(store, config):
    from concurrent.futures import ThreadPoolExecutor

    build_graph(store)
    recaller = Recaller(store, config)
    queries = ["Where do I live?", "Any gift ideas for mother?", "hello", "Nagpur weather"]

    def run(i):
        return recaller.build("st"[i % 2], [{"role": "user", "content": queries[i % 4]}], now=1.0)

    with ThreadPoolExecutor(8) as pool:
        assert all(r.messages for r in pool.map(run, range(64)))


# --- Base-level activation and importance --------------------------------------------------

DAY = 86400.0
NOW = 1000 * DAY


def asha_likes(store, *items):
    """Facts "Asha likes <object>" for (object, evidence times, importance) items; returns their ids."""
    asha = store.add_entity("s", "Asha", "person")
    likes = store.add_predicate("s", "likes", "many")
    ids = []
    for object_, times, importance in items:
        fid = store.add_fact("s", asha, likes, None, object_, times[0], [len(ids)], now=times[0],
                             importance=importance)
        for t in times[1:]:
            store.reinforce_fact(fid, [100 + len(ids)], now=t)
        ids.append(fid)
    return ids


def fact_order(store, config, query="what does Asha like"):
    return [item.id for item in Recaller(store, config)._ranked("s", query, now=NOW) if item.section == "facts"]


def test_more_evidence_outranks_a_single_observation(store, config):
    once, thrice = asha_likes(store, ("tea", [NOW], 0.5), ("coffee", [NOW] * 3, 0.5))
    assert fact_order(store, config) == [thrice, once]


def test_old_trivia_ranks_below_a_fresh_fact(store, config):
    old, fresh = asha_likes(store, ("tea", [NOW - 300 * DAY], 0.5), ("coffee", [NOW], 0.5))
    assert fact_order(store, config) == [fresh, old]


def test_surprising_evidence_raises_ranking(store, config):
    dull, surprising = asha_likes(store, ("tea", [NOW], 0.1), ("coffee", [NOW], 0.9))
    assert fact_order(store, config) == [surprising, dull]


def test_repeated_older_evidence_beats_recency_unless_ablated(store, config):
    repeated, fresh = asha_likes(store, ("tea", [NOW - 10 * DAY] * 3, 0.5), ("coffee", [NOW], 0.5))
    assert fact_order(store, config) == [repeated, fresh]  # base level: 3 * 11 ** -0.5 = 0.90 > 1
    config.enable_base_level = False  # old recency on last_used: 0.5 ** (10 / 14) = 0.61 < 1
    assert fact_order(store, config) == [fresh, repeated]


def test_events_rank_by_when_they_were_learned_and_their_importance(store, config):
    user = store.user_entity("s")
    learned_today = store.add_event("s", "user ran a marathon in Pune", NOW - 700 * DAY, True, [user], [1], now=NOW)
    learned_long_ago = store.add_event("s", "user ran a marathon in Goa", NOW, True, [user], [2], now=NOW - 300 * DAY)
    surprising = store.add_event("s", "user ran a marathon in Delhi", NOW, True, [user], [3], now=NOW, importance=1.0)

    def order():
        ranked = Recaller(store, config)._ranked("s", "user ran a marathon", now=NOW)
        return [item.id for item in ranked if item.section == "events"]

    assert order() == [surprising, learned_today, learned_long_ago]
    config.enable_base_level = False  # old: recency of occurred_at, no importance
    assert order()[-1] == learned_today


def test_memory_block_states_todays_date(store, config):
    from datetime import datetime

    build_graph(store)
    now = datetime(2026, 10, 13, 12, 0).timestamp()
    text = Recaller(store, config).build("s", [{"role": "user", "content": "hello"}], now=now).memory_text
    assert "Today is 2026-10-13." in text.splitlines()[1]
