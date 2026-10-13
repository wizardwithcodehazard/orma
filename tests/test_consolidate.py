from datetime import datetime, timezone

from smruti.sleep.consolidate import Consolidator, parse_time
from smruti.sleep.extract import parse_extraction
from smruti.store import Episode

LIVES_PUNE = {
    "entities": [{"name": "Pune", "type": "city"}],
    "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}],
}


def apply(store, config, data, ids=(1,), now=100.0, at=None, role="user"):
    episodes = [Episode(i, "s", role, "t", None, 1.0, 1, now if at is None else at, None) for i in ids]
    return Consolidator(store, config).apply("s", parse_extraction(data), episodes, now=now)


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
