from datetime import datetime, timezone

from smruti.sleep.consolidate import Consolidator, parse_time
from smruti.sleep.extract import parse_extraction

LIVES_PUNE = {
    "entities": [{"name": "Pune", "type": "city"}],
    "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}],
}


def apply(store, config, data, ids=(1,), now=100.0):
    return Consolidator(store, config).apply("s", parse_extraction(data), list(ids), now=now)


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
    apply(store, config, {"facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]})
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and fact.object_id is not None


def test_similar_entities_merge_only_when_types_match(store, config):
    config.entity_merge_threshold = 0.8
    works_at = lambda name, type_: {
        "entities": [{"name": name, "type": type_}],
        "facts": [{"subject": "user", "predicate": "works_at", "object": name, "cardinality": "one"}],
    }
    apply(store, config, works_at("Acme Corp", "company"))
    apply(store, config, works_at("Acme Corp Ltd", "Company"))
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
