import json
from datetime import datetime, timezone

import pytest

from smruti.sleep.consolidate import Consolidator, parse_time
from smruti.sleep.extract import parse_extraction
from smruti.store import Episode

LIVES_PUNE = {
    "entities": [{"name": "Pune", "type": "city"}],
    "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}],
}


def apply(store, config, data, ids=(1,), now=100.0, at=None, role="user", text=None, agreed=(), novelty=1.0):
    """text is the batch's log; by default the extraction itself, so everything in it is grounded.
    novelty is one value for every episode or a list with one per episode."""
    text = json.dumps(data) if text is None else text
    novelties = novelty if isinstance(novelty, list) else [novelty] * len(ids)
    episodes = [Episode(i, "s", role, text, None, n, 1, now if at is None else at, None) for i, n in zip(ids, novelties)]
    return Consolidator(store, config).apply("s", parse_extraction(data), episodes, now=now, agreed=agreed)


def lives(place, time="", cardinality="one", predicate="lives_in"):
    return {"facts": [{"subject": "user", "predicate": predicate, "object": place,
                       "cardinality": cardinality, "time": time}]}


def test_new_fact_is_added_and_linked_to_entities(store, config):
    report = apply(store, config, LIVES_PUNE)
    assert report.facts_added == 1
    (fact,) = store.current_facts("s")
    assert fact.text == "user lives in Pune"
    assert fact.object_id == store.find_entity("s", "pune").id


def test_restated_fact_is_reinforced(store, config):
    apply(store, config, LIVES_PUNE, ids=[1])
    report = apply(store, config, LIVES_PUNE, ids=[2])
    assert report.facts_reinforced == 1
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and fact.source_episode_ids == [1, 2]


def test_contradiction_on_one_valued_relation_closes_old_fact(store, config):
    apply(store, config, LIVES_PUNE, now=100.0)
    report = apply(store, config, {
        "entities": [{"name": "Bangalore", "type": "city"}],
        "facts": [{"subject": "user", "predicate": "lives in", "object": "Bangalore", "cardinality": "one"}],
    }, now=200.0)
    assert report.facts_closed == 1
    assert [f.text for f in store.current_facts("s")] == ["user lives in Bangalore"]
    old = [f for f in store.fact_history("s") if f.object == "Pune"][0]
    assert old.valid_to == 200.0


def test_many_valued_relation_accumulates(store, config):
    for food in ("dosa", "biryani"):
        apply(store, config, {"facts": [{"subject": "user", "predicate": "likes", "object": food, "cardinality": "many"}]})
    assert sorted(f.object for f in store.current_facts("s")) == ["biryani", "dosa"]


def test_known_entity_is_reused_when_not_reextracted(store, config):
    apply(store, config, LIVES_PUNE)
    apply(store, config, {"facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]},
          ids=[2])
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and fact.object_id is not None


def test_similar_entities_merge_only_when_types_match(store, config):
    config.entity_merge_threshold = 0.8
    works_at = lambda name, type_: {
        "entities": [{"name": name, "type": type_}],
        "facts": [{"subject": "user", "predicate": "works_at", "object": name, "cardinality": "one"}],
    }
    apply(store, config, works_at("Acme Corp", "company"))
    apply(store, config, works_at("Acme Corp Ltd", "Company"), ids=[2])
    (fact,) = store.current_facts("s")
    assert fact.object == "Acme Corp" and fact.support == 2

    apply(store, config, {"entities": [{"name": "Acme Corp Ltd", "type": "product"}]})
    assert store.find_entity("s", "Acme Corp Ltd") is not None


def test_procedures_accumulate_evidence(store, config):
    proc = {"procedures": [{"trigger": "npm run build fails with missing vite", "action": "npm install vite",
                            "polarity": "works"}]}
    apply(store, config, proc, ids=[1], role="tool")
    report = apply(store, config, proc, ids=[2], role="tool")
    assert report.procedures_reinforced == 1
    found, _ = store.nearest_procedure("s", store.embed("When npm run build fails with missing vite: npm install vite"))
    assert found.successes == 2


def test_summaries_are_capped(store, config):
    config.summary_max_tokens = 10
    apply(store, config, {"summary_updates": [{"topic": "Travel", "text": "word " * 100}]})
    summary = store.summary("s", "travel")
    assert len(summary.text) <= 35 and summary.tokens <= 10


def test_parse_time():
    assert parse_time("2026-01-01", 5.0) == datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    assert parse_time("", 5.0) == 5.0
    assert parse_time("someday", 5.0) == 5.0
    jan = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    assert parse_time("2026-01-01T00:00:00Z", 5.0) == jan
    assert parse_time("2026", 5.0) == jan
    assert parse_time("2026-03", 5.0) == datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()


def test_backdated_fact_is_inserted_closed_and_current_fact_stays(store, config):
    apply(store, config, lives("Bangalore", "2026-06-01"))
    apply(store, config, lives("Pune", "2025-01-01"), ids=[2])
    assert [f.object for f in store.current_facts("s")] == ["Bangalore"]
    old = [f for f in store.fact_history("s") if f.object == "Pune"][0]
    assert old.valid_to == datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
    assert [f.object for f in store.facts([i for i, _ in store.vec_search("facts", "s", store.embed("user lives in Pune"), 5)])] == ["Bangalore"]


def test_fact_order_in_one_extraction_does_not_matter(store, config):
    data = {"facts": [
        {"subject": "user", "predicate": "lives_in", "object": "Bangalore", "cardinality": "one", "time": "2026-06-01"},
        {"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one", "time": "2025-01-01"}]}
    apply(store, config, data)
    assert [f.object for f in store.current_facts("s")] == ["Bangalore"]


def test_default_valid_from_is_earliest_episode_time(store, config):
    apply(store, config, lives("Pune"), ids=[1, 2], now=500.0, at=50.0)
    (fact,) = store.current_facts("s")
    assert fact.valid_from == 50.0


def test_one_valued_predicate_does_not_merge_into_many_valued_neighbor(store, config):
    config.predicate_merge_threshold = 0.3
    apply(store, config, lives("Pune"))
    apply(store, config, lives("Goa", cardinality="many", predicate="visited"))
    assert sorted(p.name for p in store.predicates("s")) == ["lives_in", "visited"]


def test_same_extraction_twice_does_not_inflate_support(store, config):
    apply(store, config, LIVES_PUNE, ids=[1, 2])
    apply(store, config, LIVES_PUNE, ids=[1, 2])
    (fact,) = store.current_facts("s")
    assert fact.support == 1


def test_retry_after_partial_apply_is_a_noop_for_closed_facts(store, config):
    first = lives("Pune", "2025-01-01")
    second = lives("Bangalore", "2026-06-01")
    apply(store, config, first, ids=[1])
    apply(store, config, second, ids=[1])
    apply(store, config, first, ids=[1])
    assert len(store.fact_history("s")) == 2


def test_user_object_resolves_to_user_entity(store, config):
    apply(store, config, {"facts": [{"subject": "Asha", "predicate": "knows", "object": "User", "cardinality": "many"}]})
    (fact,) = store.current_facts("s")
    assert fact.object_id == store.user_entity("s")


def test_entity_literal_flip_reinforces(store, config):
    apply(store, config, lives("Pune"), ids=[1])
    apply(store, config, LIVES_PUNE, ids=[2])
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and len(store.fact_history("s")) == 1


PROC = {"procedures": [{"trigger": "npm run build fails with missing vite", "action": "npm install vite",
                        "polarity": "works"}]}
PROC_QUERY = "When npm run build fails with missing vite: npm install vite"


def test_procedure_evidence_is_idempotent_per_batch(store, config):
    apply(store, config, PROC, ids=[1, 2], role="tool")
    apply(store, config, PROC, ids=[1, 2], role="tool")  # retry of the same batch
    found, _ = store.nearest_procedure("s", store.embed(PROC_QUERY))
    assert found.successes == 1 and found.source_episode_ids == [1, 2]
    apply(store, config, PROC, ids=[3], role="tool")  # a new batch is new evidence
    found, _ = store.nearest_procedure("s", store.embed(PROC_QUERY))
    assert found.successes == 2 and found.source_episode_ids == [1, 2, 3]


def test_similar_procedures_in_one_batch_both_count(store, config):
    both = {"procedures": PROC["procedures"] + [dict(PROC["procedures"][0], polarity="fails")]}
    apply(store, config, both, ids=[1], role="tool")
    found, _ = store.nearest_procedure("s", store.embed(PROC_QUERY))
    assert (found.successes, found.failures) == (1, 1)


def test_close_and_insert_of_a_fact_are_atomic(store, config):
    apply(store, config, LIVES_PUNE, now=100.0)

    def boom(*args, **kwargs):
        raise RuntimeError("insert failed")

    store.add_fact = boom
    try:
        apply(store, config, lives("Bangalore"), ids=[2], now=200.0)
    except RuntimeError:
        pass
    assert [f.object for f in store.current_facts("s")] == ["Pune"]  # the close was rolled back


def test_fact_is_embedded_before_the_transaction(store, config):
    apply(store, config, LIVES_PUNE)
    in_tx = []
    real = store.embed
    store.embed = lambda text: (in_tx.append(store.conn.in_transaction), real(text))[1]
    apply(store, config, lives("Bangalore"), ids=[2], now=200.0)
    assert in_tx and not any(in_tx)


def test_recency_is_evidence_time_not_sleep_time(store, config):
    apply(store, config, {**LIVES_PUNE, **PROC}, ids=[1], now=500.0, at=50.0, role="tool")
    (fact,) = store.current_facts("s")
    assert fact.last_used == 50.0
    apply(store, config, {**LIVES_PUNE, **PROC}, ids=[2], now=900.0, at=80.0, role="tool")
    (fact,) = store.current_facts("s")
    assert fact.last_used == 80.0
    found, _ = store.nearest_procedure("s", store.embed(PROC_QUERY))
    assert found.last_used == 80.0


DAY = 86400.0


def test_move_stated_before_inferred_fact_supersedes_it(store, config):
    apply(store, config, lives("Pune"), ids=[1], now=100 * DAY, at=100 * DAY)  # no date: inferred valid_from
    apply(store, config, lives("Bangalore", "1970-03-01"), ids=[2], now=200 * DAY, at=200 * DAY)
    assert [f.object for f in store.current_facts("s")] == ["Bangalore"]
    pune = [f for f in store.fact_history("s") if f.object == "Pune"][0]
    assert pune.valid_to == 100 * DAY and not pune.valid_from_stated
    (bangalore,) = store.current_facts("s")
    assert bangalore.valid_from_stated


def test_backdating_needs_both_times_stated(store, config):
    apply(store, config, lives("Bangalore", "2026-06-01"), ids=[1])
    apply(store, config, lives("Pune"), ids=[2], at=5.0)  # inferred time earlier than a stated fact
    assert [f.object for f in store.current_facts("s")] == ["Pune"]


def test_valid_from_stated_is_migrated_onto_old_databases(config, embedder):
    from smruti.store import Store

    store = Store(config, embedder)
    store.conn.execute("ALTER TABLE facts DROP COLUMN valid_from_stated")
    store.close()
    store = Store(config, embedder)
    user = store.user_entity("s")
    lives_in = store.add_predicate("s", "lives_in", "one")
    store.add_fact("s", user, lives_in, None, "Pune", 1.0, [1], valid_from_stated=True)
    assert store.current_facts("s")[0].valid_from_stated is True
    store.close()


def test_evidence_and_old_databases_migrate(config, embedder):
    from smruti.store import Store

    store = Store(config, embedder)
    user = store.user_entity("s")
    store.add_fact("s", user, store.add_predicate("s", "lives_in", "one"), None, "Pune", 1.0, [1], now=42.0)
    store.add_event("s", "user moved to Pune", 1.0, False, [user], [1], now=42.0)
    for table, column in (("facts", "evidence_times"), ("facts", "importance"), ("events", "importance")):
        store.conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    store.close()
    store = Store(config, embedder)
    (fact,) = store.current_facts("s")
    assert fact.evidence_times == [42.0] and fact.importance == 0.5  # an old row's evidence is its last use
    assert [e.importance for e in store.batch_events("s", [1])] == [0.5]
    store.close()


def test_importance_is_the_batch_mean_novelty_and_reinforcement_keeps_the_max(store, config):
    apply(store, config, LIVES_PUNE, ids=[1, 2], at=50.0, novelty=[0.2, 0.6])
    (fact,) = store.current_facts("s")
    assert fact.importance == pytest.approx(0.4) and fact.evidence_times == [50.0]
    apply(store, config, LIVES_PUNE, ids=[3], at=80.0, novelty=0.1)
    (fact,) = store.current_facts("s")
    assert fact.importance == pytest.approx(0.4) and fact.evidence_times == [50.0, 80.0]
    apply(store, config, LIVES_PUNE, ids=[4], at=90.0, novelty=[None])  # unknown novelty counts as 0.5
    assert store.current_facts("s")[0].importance == pytest.approx(0.5)


def test_agreed_fact_records_two_evidences_at_the_batch_time(store, config):
    apply(store, config, LIVES_PUNE, ids=[1], at=50.0, agreed={0})
    assert store.current_facts("s")[0].evidence_times == [50.0, 50.0]


def test_user_names_attach_facts_to_user_entity(store, config):
    apply(store, config, {
        "user_names": ["Sahil"],
        "entities": [{"name": "Sahil", "type": "person"}, {"name": "Pune", "type": "city"}],
        "facts": [{"subject": "Sahil", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}],
    })
    user = store.user_entity("s")
    assert [(f.subject_id, f.text) for f in store.current_facts("s")] == [(user, "user lives in Pune")]
    assert store.find_entity("s", "sahil").id == user
    apply(store, config, {"facts": [{"subject": "SAHIL", "predicate": "likes", "object": "tea", "cardinality": "many"}]},
          ids=[2])
    assert {f.subject_id for f in store.current_facts("s")} == {user}


def test_procedures_need_a_tool_episode_in_the_batch(store, config):
    report = apply(store, config, PROC, ids=[1], role="user")
    assert report.procedures_added == 0 and store.nearest_procedure("s", store.embed(PROC_QUERY)) is None
    assert apply(store, config, PROC, ids=[2], role="tool").procedures_added == 1


def test_paraphrased_literal_reinforces_instead_of_contradicting(store, config):
    allergy = lambda value: {"facts": [{"subject": "user", "predicate": "is_allergic_to", "object": value,
                                        "cardinality": "one"}]}
    apply(store, config, allergy("peanut allergy"), ids=[1])
    report = apply(store, config, allergy("allergy peanut"), ids=[2])  # same words, different string
    assert report.facts_closed == 0 and report.facts_reinforced == 1
    (fact,) = store.current_facts("s")
    assert fact.support == 2


def test_fact_copied_from_context_instead_of_the_log_is_dropped(store, config):
    apply(store, config, lives("Pune"), ids=[1], text="I live in Pune")
    report = apply(store, config, {
        "entities": [{"name": "Bangalore", "type": "city"}, {"name": "Pune", "type": "city"}],
        "facts": lives("Bangalore")["facts"] + lives("Pune")["facts"],  # Pune only came from EXISTING SUMMARIES
    }, ids=[2], text="I moved to Bangalore last month")
    assert report.facts_ungrounded == 1 and report.facts_added == 1
    assert [f.object for f in store.current_facts("s")] == ["Bangalore"]


def test_grounding_uses_the_index_stemmer_and_checks_non_user_subjects(store, config):
    report = apply(store, config, {"facts": [
        {"subject": "Asha", "predicate": "is_allergic_to", "object": "peanut", "cardinality": "many"},
        {"subject": "Ravi", "predicate": "is_allergic_to", "object": "peanut", "cardinality": "many"},
    ]}, text="Asha is allergic to peanuts")
    assert report.facts_ungrounded == 1
    assert [f.text for f in store.current_facts("s")] == ["Asha is allergic to peanut"]


def test_user_names_and_entities_must_be_grounded(store, config):
    apply(store, config, {"user_names": ["Sahil", "Sam"], "entities": [{"name": "Goa", "type": "city"}]},
          text="hi, I'm Sahil")
    assert store.aliases(store.user_entity("s")) == ["sahil"]
    assert store.find_entity("s", "Goa") is None


def test_known_user_alias_subject_needs_no_mention(store, config):
    apply(store, config, {"user_names": ["Sahil"]}, ids=[1], text="I'm Sahil")
    report = apply(store, config, {"facts": [
        {"subject": "Sahil", "predicate": "likes", "object": "tea", "cardinality": "many"}]}, ids=[2], text="I like tea")
    assert report.facts_ungrounded == 0
    assert [f.text for f in store.current_facts("s")] == ["user likes tea"]


def test_grounding_can_be_disabled(store, config):
    config.enable_grounding = False
    report = apply(store, config, lives("Pune"), text="I moved to Bangalore last month")
    assert report.facts_ungrounded == 0 and [f.object for f in store.current_facts("s")] == ["Pune"]


def test_time_on_the_episode_day_is_not_stated(store, config):
    at = datetime(2026, 3, 5, 10, 0, tzinfo=timezone.utc).timestamp()
    apply(store, config, lives("Pune", "2026-03-05"), ids=[1], now=at + 3600, at=at)
    (fact,) = store.current_facts("s")
    assert fact.valid_from == at and not fact.valid_from_stated
    apply(store, config, lives("Bangalore", "2026-03-04"), ids=[2], now=at + 3600, at=at)
    bangalore = [f for f in store.fact_history("s") if f.object == "Bangalore"][0]
    assert bangalore.valid_from_stated


def test_agreed_fact_gets_one_extra_reinforcement_once(store, config):
    data = {"facts": lives("Pune")["facts"] + [
        {"subject": "user", "predicate": "likes", "object": "tea", "cardinality": "many"}]}
    report = apply(store, config, data, ids=[1, 2], agreed={0})
    assert report.facts_agreed == 1 and report.facts_added == 2
    support = {f.object: f.support for f in store.current_facts("s")}
    assert support == {"Pune": 2, "tea": 1}
    apply(store, config, data, ids=[1, 2], agreed={0})  # retry of the same batch
    assert {f.object: f.support for f in store.current_facts("s")} == support
    apply(store, config, lives("Pune"), ids=[3], agreed={0})  # agreed restatement in a new batch
    assert [f.support for f in store.current_facts("s") if f.object == "Pune"] == [4]


MARCH_10 = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc).timestamp()
GOA_TRIP = {
    "entities": [{"name": "Asha", "type": "person"}, {"name": "Goa", "type": "city"}],
    "events": [{"gist": "user went to Goa with Asha", "time": "2026-03-07", "participants": ["user", "Asha", "Goa"]}],
}


def trip(gist="user went to Goa with Asha", time="", participants=("user", "Asha")):
    return {"events": [{"gist": gist, "time": time, "participants": list(participants)}]}


def test_event_is_stored_with_its_date_and_participants(store, config):
    report = apply(store, config, GOA_TRIP, ids=[1, 2], now=MARCH_10 + 3600, at=MARCH_10)
    assert report.events_added == 1
    (event,) = store.batch_events("s", [1, 2])
    assert event.gist == "user went to Goa with Asha" and event.text == "2026-03-07: user went to Goa with Asha"
    assert event.occurred_at == datetime(2026, 3, 7, tzinfo=timezone.utc).timestamp() and event.occurred_stated
    assert event.source_episode_ids == [1, 2] and event.created_at == MARCH_10
    expected = {store.user_entity("s"), store.find_entity("s", "Asha").id, store.find_entity("s", "Goa").id}
    assert set(store.event_participants(event.id)) == expected
    assert store.stats("s")["events"] == 1


def test_event_without_a_stated_time_happened_when_the_batch_began(store, config):
    apply(store, config, trip(time="2026-03-10"), ids=[1, 2], now=MARCH_10 + 3600, at=MARCH_10)  # the line's own day
    (event,) = store.batch_events("s", [1, 2])
    assert event.occurred_at == MARCH_10 and not event.occurred_stated


def test_events_with_ungrounded_or_no_participants_are_dropped(store, config):
    report = apply(store, config, {"events": trip(participants=["user", "Ravi"])["events"]
                                   + trip(gist="user drove to Goa", participants=[])["events"]
                                   + trip(gist="user went to Goa", participants=["user", "Goa"])["events"]},
                   text="Yesterday I went to Goa with my sister")
    assert report.events_ungrounded == 2 and report.events_added == 1
    assert [e.gist for e in store.batch_events("s", [1])] == ["user went to Goa"]


def test_known_user_alias_participant_needs_no_mention(store, config):
    apply(store, config, {"user_names": ["Sahil"]}, ids=[1], text="I'm Sahil")
    report = apply(store, config, trip(gist="Sahil went to Goa", participants=["Sahil", "Goa"]), ids=[2],
                   text="I went to Goa")
    assert report.events_added == 1
    (event,) = store.batch_events("s", [2])
    assert store.user_entity("s") in store.event_participants(event.id)


def test_retried_batch_does_not_duplicate_events(store, config):
    apply(store, config, GOA_TRIP, ids=[1, 2])
    report = apply(store, config, GOA_TRIP, ids=[1, 2])
    assert report.events_added == 0 and len(store.batch_events("s", [1, 2])) == 1
    apply(store, config, trip(gist="user cooked dinner with Asha"), ids=[1, 2])  # a different event of the batch
    assert len(store.batch_events("s", [1, 2])) == 2


def test_events_can_be_disabled(store, config):
    config.enable_events = False
    report = apply(store, config, GOA_TRIP)
    assert report.events_added == 0 and store.stats("s")["events"] == 0


def test_event_importance_is_the_batch_mean_novelty(store, config):
    apply(store, config, GOA_TRIP, ids=[1, 2], now=MARCH_10 + 3600, at=MARCH_10, novelty=[0.9, 0.3])
    (event,) = store.batch_events("s", [1, 2])
    assert event.importance == pytest.approx(0.6)
