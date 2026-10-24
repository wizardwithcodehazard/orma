import json
from pathlib import Path

import pytest

from eval.run import load_cases, overrides, run_case, summarize, table
from eval.score import answer_passes, finds, label, state_matches
from smruti.recall import SearchResult

ROOT = Path(__file__).resolve().parent.parent


class FakeState:
    def __init__(self, subject, attribute, value):
        self.subject, self.attribute, self.value = subject, attribute, value
        self.text = f"{subject} · {attribute}: {value}"


class FakeSentence:
    def __init__(self, text):
        self.text = text


def test_dev_and_test_cases_load_and_do_not_overlap():
    _, dev = load_cases(ROOT / "eval" / "cases.json")
    tz, test = load_cases(ROOT / "eval" / "test_cases.json")
    assert tz == "Asia/Kolkata" and sum(len(c["questions"]) for c in dev) == 10
    assert sum(len(c["questions"]) for c in test) == 10
    assert not {c["id"] for c in dev} & {c["id"] for c in test}


def test_unknown_case_ids_are_rejected():
    with pytest.raises(SystemExit):
        load_cases(ROOT / "eval" / "cases.json", {"no-such-case"})


def test_state_matching_is_lenient_on_wording_and_strict_on_holder():
    spec = {"about": ["user"], "terms": ["bangalore"]}
    assert state_matches(FakeState("user", "home_city", "Bangalore"), spec)
    assert state_matches(FakeState("User", "lives_in", "bangalore"), spec)
    assert not state_matches(FakeState("Asha", "home_city", "Bangalore"), spec)
    either = {"any": [spec, {"about": ["asha"], "terms": ["delhi"]}]}
    assert state_matches(FakeState("Asha", "city", "Delhi"), either)
    assert label(either) == "user: bangalore | asha: delhi"


def test_finds_looks_in_top_passages_and_states():
    result = SearchResult(states=[FakeState("user", "home_city", "Pune")],
                          passages=[(FakeSentence("My landlord is Mr. Iyer."), 1.0)])
    assert finds(result, ["iyer"]) == (True, "My landlord is Mr. Iyer.")
    assert finds(result, ["pune"])[0] and not finds(result, ["mumbai"])[0]


def test_answers_match_whole_words():
    assert answer_passes("No, you quit last week.", [["no", "quit"]], [])
    assert not answer_passes("I know you love it", [["no"]], [])
    assert not answer_passes("Visit Pune or Nashik", [["nashik"]], [["pune"]])
    assert answer_passes("Anything at all", [], [["chicken"]])


def test_overrides_are_typed_by_the_config():
    assert overrides(["enable_semantic=false", "budget_tokens=2000", "use_weight=0.5", "timezone=UTC"]) == {
        "enable_semantic": False, "budget_tokens": 2000, "use_weight": 0.5, "timezone": "UTC"}
    with pytest.raises(SystemExit):
        overrides(["no_such_field=1"])


def test_a_case_runs_end_to_end_with_fakes(tmp_path, fake_llm, embedder):
    case = {
        "id": "sister",
        "sessions": [{"at": "2026-10-05T18:00:00", "turns": [
            "My sister Asha lives in Delhi.",
            {"role": "document", "speaker": "notes.md", "text": "Asha is allergic to peanuts."}]}],
        "expect": {"current": [{"about": ["asha"], "terms": ["delhi"]}],
                   "absent": [{"about": ["user"], "terms": ["delhi"]}],
                   "user_aliases": ["Sahil"]},
        "finds": [{"q": "cookies for my sister", "terms": ["peanuts"]}],
        "questions": [{"q": "Cookie ideas for my sister?", "must": [["peanut-free"]]}],
    }
    extraction = {"states": [{"s": 1, "about": "Asha", "attribute": "home_city", "value": "Delhi", "ended": False,
                              "since": ""}], "events": []}
    llm = fake_llm([extraction], chat_reply="Bake peanut-free oat cookies.")
    lines = []
    score, answers = run_case(case, "UTC", {"embedding_dim": embedder.dim}, llm, "m", embedder, samples=2,
                              baseline=True, workdir=tmp_path, log=lines.append)
    by_kind = {c.kind: c.passed for c in score.checks}
    assert by_kind == {"current": True, "absent": True, "alias": False, "finds": True, "question": True}
    assert [a["memory"] for a in answers] == [True, True, False] and all(a["passed"] for a in answers)
    assert "FAIL alias" in "\n".join(lines)
    assert "Asha is allergic to peanuts" in json.dumps(llm.calls[-2])  # the document reached the model

    summary = summarize([score], answers)
    assert summary["states_found"] == (1, 1) and summary["retrieval"] == (1, 1)
    assert "| states_found | 1/1 |" in table([("x", summary)])
