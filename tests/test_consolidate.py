from datetime import datetime, timezone

from smruti.sleep.consolidate import Consolidator, parse_time
from smruti.sleep.extract import parse_extraction
from smruti.store import Episode

LIVES_PUNE = {
    "entities": [{"name": "Pune", "type": "city"}],
    "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}],
}


def apply(store, config, data, ids=(1,), now=100.0, at=None):
    episodes = [Episode(i, "s", "user", "t", None, 1.0, 1, now if at is None else at, None) for i in ids]
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
    apply(store, config, proc)
    report = apply(store, config, proc)
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
