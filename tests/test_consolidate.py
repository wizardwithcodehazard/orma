from datetime import datetime, timezone

import pytest

from orma.encode import Encoder
from orma.sleep.consolidate import Consolidator
from orma.sleep.extract import parse_extraction

THURSDAY = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc).timestamp()


def keep(store, config, *lines, at=THURSDAY):
    """Adds each (text, role) line as a record; returns all their sentences, in order."""
    enc = Encoder(store, config)
    ids = []
    for i, (text, role) in enumerate(lines):
        result = enc.encode("s", role, text, now=at + i, speaker="user" if role in ("user", "note") else role)
        ids.append(result.episode_id)
    return store.episode_sentences(ids)


def apply(store, config, data, sentences, now=None):
    return Consolidator(store, config).apply("s", parse_extraction(data), sentences, now=now or THURSDAY + 600)


def state(s, attribute, value, about="user", ended=False, since=""):
    return {"s": s, "about": about, "attribute": attribute, "value": value, "ended": ended, "since": since}


def current(store):
    return sorted(st.text for st in store.current_states("s"))


# --- citation and holders ------------------------------------------------------------

def test_a_state_cites_its_line(store, config):
    (line,) = keep(store, config, ("I live in Pune.", "user"))
    report = apply(store, config, {"states": [state(line.id, "home_city", "Pune")]}, [line])
    assert report.states_added == 1
    (st,) = store.current_states("s")
    assert (st.text, st.sentence_id, st.evidence, st.valid_from) == ("user · home city: Pune", line.id, [line.id],
                                                                      line.created_at)


def test_uncited_and_assistant_lines_are_not_evidence(store, config):
    lines = keep(store, config, ("I live in Pune.", "user"), ("You must love Pune's weather!", "assistant"))
    report = apply(store, config, {"states": [state(999, "home_city", "Pune"),
                                              state(lines[1].id, "weather_opinion", "love")]}, lines)
    assert report.uncited == 2 and store.current_states("s") == []


def test_first_person_means_the_user_only_in_the_users_own_lines(store, config):
    lines = keep(store, config, ("I work at Razorpay.", "user"), ("I am the release notes for v2.", "document"))
    report = apply(store, config, {"states": [state(lines[0].id, "employer", "Razorpay", about="I"),
                                              state(lines[1].id, "version", "v2", about="I")]}, lines)
    assert current(store) == ["user · employer: Razorpay"] and report.unresolved == 1


def test_a_pronoun_carries_over_the_last_name_before_it(store, config):
    lines = keep(store, config, ("My sister Asha lives in Delhi.", "user"), ("She works at Infosys.", "user"))
    apply(store, config, {"states": [state(lines[0].id, "home_city", "Delhi", about="Asha"),
                                     state(lines[1].id, "employer", "Infosys", about="she")]}, lines)
    assert current(store) == ["Asha · employer: Infosys", "Asha · home city: Delhi"]


def test_a_pronoun_carries_over_a_name_even_when_no_memory_was_made_of_it(store, config):
    lines = keep(store, config, ("My friend Rohan called today.", "user"), ("He moved to Chennai.", "user"))
    data = {"states": [state(lines[1].id, "home_city", "Chennai", about="he")],
            "events": [{"s": lines[0].id, "what": "Rohan called", "when": "", "who": ["Rohan"]}]}
    apply(store, config, data, lines)
    assert current(store) == ["Rohan · home city: Chennai"]


def test_a_pronoun_with_nobody_before_it_is_unresolved(store, config):
    (line,) = keep(store, config, ("She moved to Chennai.", "user"))
    report = apply(store, config, {"states": [state(line.id, "home_city", "Chennai", about="she")]}, [line])
    assert report.unresolved == 1 and store.current_states("s") == []


def test_a_name_neither_in_the_batch_nor_known_is_unresolved(store, config):
    (line,) = keep(store, config, ("She moved to Chennai.", "user"))
    report = apply(store, config, {"states": [state(line.id, "home_city", "Chennai", about="Meera")]}, [line])
    assert report.unresolved == 1


def test_the_users_name_becomes_an_alias_only_when_they_say_it(store, config):
    (line,) = keep(store, config, ("Hi, I'm Sahil.", "user"))
    report = apply(store, config, {"user_name": "Sahil"}, [line])
    assert report.aliases_added == 1 and store.find_entity("s", "Sahil").name == "user"
    report = apply(store, config, {"user_name": "Priya"}, [line])
    assert report.aliases_rejected == 1 and store.find_entity("s", "Priya") is None


def test_an_existing_person_never_becomes_the_users_name(store, config):
    store.add_entity("s", "Asha", "thing")
    (line,) = keep(store, config, ("Asha said hi to me.", "user"))
    report = apply(store, config, {"user_name": "Asha"}, [line])
    assert report.aliases_rejected == 1 and store.find_entity("s", "Asha").name == "Asha"


# --- grounding -----------------------------------------------------------------------

def test_a_value_must_be_in_the_line_it_cites(store, config):
    """A value copied from the known context into a line that does not say it is dropped."""
    (line,) = keep(store, config, ("I moved to Bangalore.", "user"))
    report = apply(store, config, {"states": [state(line.id, "home_city", "Pune")]}, [line])
    assert report.ungrounded == 1 and store.current_states("s") == []


class SynonymEmbedder:
    """FakeEmbedder that reads "mom" as "mother", like a real model scoring the pair ~0.93."""

    def __init__(self, base):
        self.base, self.dim = base, base.dim

    def embed(self, texts):
        return self.base.embed([" ".join("mother" if w.lower() == "mom" else w for w in t.split()) for t in texts])


def test_a_paraphrased_value_is_grounded(config, embedder):
    from orma.store import Store

    store = Store(config, SynonymEmbedder(embedder))
    (line,) = keep(store, config, ("I am staying with mom this week.", "user"))
    report = apply(store, config, {"states": [state(line.id, "staying_with", "mother")]}, [line])
    assert report.states_added == 1
    store.close()


def test_grounding_can_be_disabled(store, config):
    config.enable_grounding = False
    (line,) = keep(store, config, ("I moved to Bangalore.", "user"))
    apply(store, config, {"states": [state(line.id, "home_city", "Pune")]}, [line])
    assert current(store) == ["user · home city: Pune"]


# --- time ----------------------------------------------------------------------------

def test_a_new_value_closes_the_old_one(store, config):
    first = keep(store, config, ("I live in Pune.", "user"))
    apply(store, config, {"states": [state(first[0].id, "home_city", "Pune")]}, first)
    later = keep(store, config, ("I moved to Bangalore.", "user"), at=THURSDAY + 86400)
    report = apply(store, config, {"states": [state(later[0].id, "home_city", "Bangalore")]}, later)
    assert report.states_closed == 1 and current(store) == ["user · home city: Bangalore"]
    pune = next(st for st in store.state_history("s") if st.value == "Pune")
    assert pune.valid_to == later[0].created_at and pune.ended_by == later[0].id


def test_an_attribute_named_differently_merges_by_meaning(store, config):
    config.attribute_merge_threshold = 0.5  # FakeEmbedder: "home city" vs "home city name" share words
    first = keep(store, config, ("I live in Pune.", "user"))
    apply(store, config, {"states": [state(first[0].id, "home_city", "Pune")]}, first)
    later = keep(store, config, ("I moved to Bangalore.", "user"), at=THURSDAY + 86400)
    apply(store, config, {"states": [state(later[0].id, "home_city_name", "Bangalore")]}, later)
    assert current(store) == ["user · home city: Bangalore"]


def test_the_same_value_again_is_one_more_use(store, config):
    lines = keep(store, config, ("I live in Pune.", "user"), ("Still in Pune, yes.", "user"))
    report = apply(store, config, {"states": [state(lines[0].id, "home_city", "Pune"),
                                              state(lines[1].id, "home_city", "pune")]}, lines)
    assert (report.states_added, report.states_reinforced) == (1, 1)
    (st,) = store.current_states("s")
    assert st.evidence == [lines[0].id, lines[1].id] and len(st.evidence_times) == 2


def test_ended_closes_the_matching_value_only(store, config):
    lines = keep(store, config, ("I drink coffee.", "user"), ("I quit coffee.", "user"), ("I quit tea.", "user"))
    report = apply(store, config, {"states": [state(lines[0].id, "daily_drink", "coffee"),
                                              state(lines[2].id, "daily_drink", "tea", ended=True),
                                              state(lines[1].id, "daily_drink", "coffee", ended=True)]}, lines)
    assert report.states_closed == 1 and current(store) == []
    (coffee,) = store.state_history("s")
    assert coffee.ended_by == lines[1].id


def test_date_words_set_when_a_state_began(store, config):
    pytest.importorskip("dateparser")
    (line,) = keep(store, config, ("I moved to Bangalore last month.", "user"))
    apply(store, config, {"states": [state(line.id, "home_city", "Bangalore", since="last month")]}, [line])
    (st,) = store.current_states("s")
    assert st.valid_from_stated and datetime.fromtimestamp(st.valid_from, timezone.utc).date().isoformat() == "2026-09-15"


def test_today_is_not_a_stated_time(store, config):
    (line,) = keep(store, config, ("I moved to Bangalore today.", "user"))
    apply(store, config, {"states": [state(line.id, "home_city", "Bangalore", since="today")]}, [line])
    assert not store.current_states("s")[0].valid_from_stated


def test_an_earlier_dated_value_is_history_not_a_change(store, config):
    lines = keep(store, config, ("I have lived in Bangalore since 2025.", "user"), ("In 2019 I lived in Pune.", "user"))
    apply(store, config, {"states": [state(lines[0].id, "home_city", "Bangalore", since="2025"),
                                     state(lines[1].id, "home_city", "Pune", since="2019")]}, lines)
    assert current(store) == ["user · home city: Bangalore"]
    pune = next(st for st in store.state_history("s") if st.value == "Pune")
    assert pune.valid_to == datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()


def test_applying_the_same_extraction_twice_changes_nothing(store, config):
    lines = keep(store, config, ("I live in Pune.", "user"), ("We went to Goa last Saturday.", "user"))
    data = {"states": [state(lines[0].id, "home_city", "Pune")],
            "events": [{"s": lines[1].id, "what": "user went to Goa", "when": "last Saturday", "who": ["user"]}]}
    apply(store, config, data, lines)
    again = apply(store, config, data, lines)
    assert (again.states_added, again.states_reinforced, again.events_added) == (0, 0, 0)
    assert len(store.state_history("s")) == 1 and len(store.space_events("s")) == 1


# --- events --------------------------------------------------------------------------

def test_events_are_dated_from_their_words_and_linked_to_who_took_part(store, config):
    pytest.importorskip("dateparser")
    lines = keep(store, config, ("Last Saturday I went trekking with my sister Asha.", "user"))
    report = apply(store, config, {"events": [{"s": lines[0].id, "what": "user went trekking with Asha",
                                               "when": "last Saturday", "who": ["user", "Asha", "Meera"]}]}, lines)
    (event,) = store.space_events("s")
    assert report.events_added == 1 and event.occurred_stated
    assert datetime.fromtimestamp(event.occurred_at, timezone.utc).date().isoformat() == "2026-10-10"
    names = {store.entity(i).name for i in store.event_participants(event.id)}
    assert names == {"user", "Asha"}  # Meera is neither in the batch nor known
