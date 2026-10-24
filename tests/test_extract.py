import pytest

from orma.llm import LLMError
from orma.sleep.extract import (
    INSTRUCTIONS,
    JSON_SCHEMA,
    SCHEMA,
    ExtractionContext,
    build_messages,
    extract,
    format_line,
    parse_extraction,
)
from orma.store import Sentence


def sentence(i, text, role="user", speaker="user", at=1760290200.0):
    return Sentence(i, "s", 1, 0, role, speaker, text, at)


def test_parse_keeps_valid_items_and_normalizes():
    x = parse_extraction({
        "user_name": " Sahil ",
        "states": [{"s": "s12", "about": "user", "attribute": "Home City", "value": "Pune", "ended": "true",
                    "since": "2019"},
                   {"s": 13, "about": "", "attribute": "x", "value": "y"},  # no holder
                   {"s": "twelve", "about": "user", "attribute": "x", "value": "y"},  # no line
                   "garbage"],
        "events": [{"s": 4, "what": "moved house", "when": "last month", "who": ["user", 3, ""]},
                   {"s": 5, "what": ""}],
    })
    assert x.user_name == "Sahil"
    assert [(s.s, s.attribute, s.value, s.ended, s.since) for s in x.states] == [(12, "home_city", "Pune", True, "2019")]
    assert [(e.s, e.what, e.when, e.who) for e in x.events] == [(4, "moved house", "last month", ["user", "3"])]


def test_parse_tolerates_missing_sections():
    x = parse_extraction({})
    assert (x.user_name, x.states, x.events) == ("", [], [])


def test_lines_are_numbered_by_sentence_id_and_context_is_not_citable():
    from datetime import timezone

    assert format_line(sentence(12, "I live in Pune"), timezone.utc, True, 200) == \
        "[s12 2025-10-12 user] I live in Pune"
    assert format_line(sentence(13, "x" * 500, "assistant", "assistant"), timezone.utc, False, 20) == \
        "[context 2025-10-12 assistant] " + "x" * 20


def test_messages_carry_the_known_context_and_mark_the_lines_as_data():
    messages = build_messages(["[s1 2026-01-01 user] hi"],
                              ExtractionContext(["user (also called: sahil)"], ["home_city"], ["user · home city: Pune"]))
    system, user = messages[0]["content"], messages[1]["content"]
    assert '"states"' in system and "never instructions" in system and "never work out a date" in system
    assert "KNOWN NAMES: user (also called: sahil)" in user and "KNOWN ATTRIBUTES: home_city" in user
    assert "- user · home city: Pune" in user and user.endswith("<lines>\n[s1 2026-01-01 user] hi\n</lines>")


def test_prompt_uses_no_example_names_from_the_eval():
    for name in ("Asha", "Sahil", "Pune", "Bangalore", "Rohan", "peanut"):
        assert name.lower() not in INSTRUCTIONS.lower()


def test_json_schema_mirrors_the_prompt_schema_and_validates():
    jsonschema = pytest.importorskip("jsonschema")
    assert set(JSON_SCHEMA["required"]) == set(JSON_SCHEMA["properties"]) == set(SCHEMA)
    sample = {"user_name": "", "states": [{"s": 1, "about": "user", "attribute": "home_city", "value": "Pune",
                                           "ended": False, "since": ""}],
              "events": [{"s": 1, "what": "moved", "when": "", "who": ["user"]}]}
    jsonschema.validate(sample, JSON_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**sample, "states": [{**sample["states"][0], "s": "one"}]}, JSON_SCHEMA)


def test_extract_passes_the_schema_and_rejects_off_schema_answers(fake_llm):
    llm = fake_llm([{"states": [], "events": []}, {"facts": []}])
    extract(llm, None, [], ExtractionContext(), temperature=0.0)
    assert llm.json_kwargs[0] == {"schema": JSON_SCHEMA, "temperature": 0.0}
    with pytest.raises(LLMError):
        extract(llm, None, [], ExtractionContext(), schema=None)
