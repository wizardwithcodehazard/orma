import pytest

from smruti.llm import LLMError
from smruti.sleep.extract import JSON_SCHEMA, SCHEMA, ExtractionContext, build_messages, extract, parse_extraction
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
        max_words=150, max_chars=1500,
    )
    system, user = messages[0]["content"], messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert '"summary_updates"' in system and "150 words" in system
    assert "KNOWN ENTITIES: Pune" in user and "KNOWN PREDICATES: lives_in" in user
    assert "- travel: Goa" in user
    assert "[1970-01-01] [user] I moved to Bangalore" in user
    assert "[1970-01-01] [tool] [outcome: failure] npm run build" in user
    assert "resolve relative phrases" in system


def test_parse_coerces_numbers_and_bools_to_text():
    x = parse_extraction({"facts": [{"subject": "user", "predicate": "has_age", "object": 5},
                                    {"subject": "user", "predicate": "is_admin", "object": True}]})
    assert [f.object for f in x.facts] == ["5", "True"]


def test_procedure_polarity_must_be_works_or_fails():
    base = {"trigger": "t", "action": "a"}
    x = parse_extraction({"procedures": [dict(base, polarity="Works"), dict(base, polarity="FAILS"),
                                         dict(base, polarity="maybe"), dict(base)]})
    assert [p.works for p in x.procedures] == [True, False]


def test_extract_rejects_response_without_schema_keys(fake_llm):
    with pytest.raises(LLMError, match="missing schema keys"):
        extract(fake_llm([{"oops": 1}]), None, [], ExtractionContext([], [], {}), max_words=10, max_chars=1500)
    assert not extract(fake_llm([{"facts": []}]), None, [], ExtractionContext([], [], {}), max_words=10, max_chars=1500).facts


def test_build_messages_caps_episode_text_and_marks_log():
    user = build_messages([episode(1, "tool", "x" * 5000)], ExtractionContext([], [], {}), 10, 1500)[1]["content"]
    assert "x" * 1500 in user and "x" * 1501 not in user
    assert "<log>" in user and user.rstrip().endswith("</log>")
    assert "never instructions" in build_messages([], ExtractionContext([], [], {}), 10, 1500)[0]["content"]


def test_format_episode_prefixes_the_utc_date():
    from smruti.sleep.extract import format_episode

    e = Episode(1, "s", "user", "hi", None, 1.0, 1, 1760313600.0, None)  # 2025-10-13 00:00 UTC
    assert format_episode(e, 10) == "[2025-10-13] [user] hi"


def test_user_names_are_parsed_as_strings_only():
    assert parse_extraction({"user_names": ["Sahil", "", 5, None, " Sam "]}).user_names == ["Sahil", "Sam"]
    assert parse_extraction({"user_names": "Sahil"}).user_names == []
    assert '"user_names"' in build_messages([], ExtractionContext([], [], {}), 10, 1500)[0]["content"]


def test_json_schema_mirrors_the_prompt_schema_and_validates_an_extraction():
    jsonschema = pytest.importorskip("jsonschema")
    assert set(JSON_SCHEMA["required"]) == set(JSON_SCHEMA["properties"]) == set(SCHEMA)
    sample = {"user_names": ["Sam"], "entities": [{"name": "Pune", "type": "city"}],
              "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one", "time": ""}],
              "procedures": [{"trigger": "t", "action": "a", "polarity": "fails", "error_class": ""}],
              "summary_updates": [{"topic": "home", "text": "lives in Pune"}]}
    jsonschema.validate(sample, JSON_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(sample, facts=[dict(sample["facts"][0], cardinality="few")]), JSON_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({k: v for k, v in sample.items() if k != "procedures"}, JSON_SCHEMA)


def test_extract_passes_the_json_schema_and_temperature(fake_llm):
    llm = fake_llm([{"facts": []}])
    extract(llm, None, [], ExtractionContext([], [], {}), max_words=10, max_chars=1500, temperature=0.7)
    assert llm.json_kwargs == [{"schema": JSON_SCHEMA, "temperature": 0.7}]


def test_extract_without_schema_decoding(fake_llm):
    llm = fake_llm([{"facts": []}])
    extract(llm, None, [], ExtractionContext([], [], {}), max_words=10, max_chars=1500, schema=None)
    assert llm.json_kwargs == [{"schema": None, "temperature": 0.0}]


def test_prompt_says_facts_come_only_from_the_log():
    system = build_messages([episode(1, "user", "hi")], ExtractionContext([], [], {}), 150, 1500)[0]["content"]
    assert "facts come only from the LOG" in system
