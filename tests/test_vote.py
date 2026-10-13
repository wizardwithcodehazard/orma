from smruti.sleep.extract import ExtractedFact, ExtractedProcedure, Extraction
from smruti.sleep.vote import vote


def fact(subject, predicate, obj, cardinality="many"):
    return ExtractedFact(subject, predicate, obj, cardinality, "")


def embedder_fn(embedder):
    return lambda text: embedder.embed([text])[0]


def test_facts_agree_across_predicate_wording(embedder):
    first = Extraction(entities={"Priya": "person"}, facts=[fact("user", "has_sister", "Priya")])
    second = Extraction(facts=[fact("user", "has_older_sister", "Priya")])
    agreed, unconfirmed = vote([first, second], embedder_fn(embedder), 0.8)
    assert agreed.facts == first.facts and unconfirmed == []
    assert agreed.entities == {"Priya": "person"}  # referenced by an agreed fact


def test_fact_seen_in_one_pass_only_is_unconfirmed(embedder):
    pune = fact("user", "lives_in", "Pune", "one")
    first = Extraction(entities={"Pune": "city", "Tea": "drink"}, facts=[pune, fact("user", "likes", "Tea")])
    second = Extraction(entities={"tea": "drink"}, facts=[fact("user", "lives_in", "Pune", "one")])
    agreed, unconfirmed = vote([first, second], embedder_fn(embedder), 0.8)
    assert agreed.facts == [pune]
    assert unconfirmed == [fact("user", "likes", "Tea")]
    assert agreed.entities == {"Pune": "city", "Tea": "drink"}  # Tea is still named by pass 2


def test_unreferenced_entity_named_only_in_pass_one_is_dropped(embedder):
    first = Extraction(entities={"Goa": "city"}, facts=[fact("user", "visited", "Goa")])
    agreed, unconfirmed = vote([first, Extraction()], embedder_fn(embedder), 0.8)
    assert agreed.entities == {} and len(unconfirmed) == 1


def test_user_names_are_a_case_insensitive_intersection(embedder):
    first = Extraction(user_names=["Sam", "Sammy"])
    second = Extraction(user_names=["sam", "Samuel"])
    agreed, _ = vote([first, second], embedder_fn(embedder), 0.8)
    assert agreed.user_names == ["Sam"]


def test_procedures_and_summaries_come_from_pass_one(embedder):
    first = Extraction(procedures=[ExtractedProcedure("t", "a", True, "")], summaries={"home": "Pune"})
    agreed, _ = vote([first, Extraction()], embedder_fn(embedder), 0.8)
    assert agreed.procedures == first.procedures and agreed.summaries == first.summaries


def test_single_extraction_is_all_agreed(embedder):
    first = Extraction(user_names=["Sam"], entities={"Goa": "city"}, facts=[fact("user", "visited", "Goa")])
    agreed, unconfirmed = vote([first], embedder_fn(embedder), 0.8)
    assert agreed == first and unconfirmed == []
