from smruti.sleep.extract import ExtractionContext, build_messages, parse_extraction
from smruti.store import Episode


def episode(i, role, text, outcome=None):
    return Episode(i, "s", role, text, outcome, 1.0, 1, float(i), None)


def test_parse_keeps_valid_items_and_drops_malformed():
    x = parse_extraction({
        "entities": [{"name": "Pune", "type": "city"}, {"type": "x"}, "junk"],
        "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"},
                  {"subject": "user", "predicate": "likes"}],
        "procedures": [{"trigger": "build fails", "action": "npm install", "polarity": "fails", "error_class": "dep"}],
        "summary_updates": [{"topic": "Travel", "text": "Planning Goa"}, {"topic": "", "text": "x"}],
    })
    assert x.entities == {"Pune": "city"}
    assert [(f.subject, f.predicate, f.object, f.cardinality) for f in x.facts] == [("user", "lives_in", "Pune", "one")]
    assert x.procedures[0].works is False and x.procedures[0].error_class == "dep"
    assert x.summaries == {"travel": "Planning Goa"}


def test_parse_tolerates_missing_sections():
    x = parse_extraction({})
    assert not x.entities and not x.facts and not x.procedures and not x.summaries


def test_unknown_cardinality_defaults_to_many():
    x = parse_extraction({"facts": [{"subject": "user", "predicate": "likes", "object": "tea", "cardinality": "?"}]})
    assert x.facts[0].cardinality == "many"


def test_build_messages_includes_schema_context_and_outcomes():
    messages = build_messages(
        [episode(1, "user", "I moved to Bangalore"), episode(2, "tool", "npm run build", "failure")],
        ExtractionContext(["Pune"], ["lives_in"], {"travel": "Goa"}),
        max_words=150,
    )
    system, user = messages[0]["content"], messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert '"summary_updates"' in system and "150 words" in system
    assert "KNOWN ENTITIES: Pune" in user and "KNOWN PREDICATES: lives_in" in user
    assert "- travel: Goa" in user
    assert "[user] I moved to Bangalore" in user
    assert "[tool] [outcome: failure] npm run build" in user
