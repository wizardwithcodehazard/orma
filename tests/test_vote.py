from smruti.sleep.extract import ExtractedEvent, ExtractedFact, ExtractedProcedure, Extraction
from smruti.sleep.vote import vote


def fact(subject, predicate, obj, cardinality="many"):
    return ExtractedFact(subject, predicate, obj, cardinality, "")


def embedder_fn(embedder):
    return lambda text: embedder.embed([text])[0]


def test_facts_agree_across_predicate_wording(embedder):
    first = Extraction(entities={"Priya": "person"}, facts=[fact("user", "has_sister", "Priya")])
    second = Extraction(facts=[fact("user", "has_older_sister", "Priya")])
    extraction, agreed = vote([first, second], embedder_fn(embedder), 0.8)
    assert extraction.facts == first.facts and agreed == {0}


def test_every_first_pass_fact_is_kept_and_only_replayed_ones_are_agreed(embedder):
    pune = fact("user", "lives_in", "Pune", "one")
    first = Extraction(entities={"Pune": "city", "Tea": "drink"}, facts=[fact("user", "likes", "Tea"), pune])
    second = Extraction(facts=[fact("user", "lives_in", "Pune", "one")])
    extraction, agreed = vote([first, second], embedder_fn(embedder), 0.8)
    assert extraction.facts == first.facts and agreed == {1}
    assert extraction.entities == first.entities


def test_user_names_are_a_case_insensitive_union(embedder):
    first = Extraction(user_names=["Sam", "Sammy"])
    second = Extraction(user_names=["sam", "Samuel"])
    extraction, _ = vote([first, second], embedder_fn(embedder), 0.8)
    assert extraction.user_names == ["Sam", "Sammy", "Samuel"]


def test_procedures_and_summaries_come_from_pass_one(embedder):
    first = Extraction(procedures=[ExtractedProcedure("t", "a", True, "")], summaries={"home": "Pune"})
    extraction, agreed = vote([first, Extraction()], embedder_fn(embedder), 0.8)
    assert extraction.procedures == first.procedures and extraction.summaries == first.summaries
    assert agreed == set()


def test_single_extraction_has_no_agreement(embedder):
    first = Extraction(user_names=["Sam"], entities={"Goa": "city"}, facts=[fact("user", "visited", "Goa")])
    extraction, agreed = vote([first], embedder_fn(embedder), 0.8)
    assert extraction == first and agreed == set()


def test_events_come_from_pass_one(embedder):
    first = Extraction(events=[ExtractedEvent("user went to Goa", "2026-03-07", ["user", "Goa"])])
    second = Extraction(events=[ExtractedEvent("user visited Goa", "", ["user"])])
    extraction, _ = vote([first, second], embedder_fn(embedder), 0.8)
    assert extraction.events == first.events
