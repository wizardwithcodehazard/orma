from datetime import datetime, timezone

from orma.encode import Encoder
from orma.recall import Recaller
from orma.recall.pack import FRAMING

DAY = 86400.0
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()


def keep(store, config, text, role="user", now=NOW - DAY, space="s"):
    return Encoder(store, config).encode(space, role, text, now=now, speaker="user" if role == "user" else role)


def ask(store, config, query, now=NOW, budget=None, history=()):
    messages = [*history, {"role": "user", "content": query}]
    return Recaller(store, config).build("s", messages, budget, now=now)


def lines(result, section):
    return [i.text for i in result.used if i.section == section]


def test_what_was_added_is_found_before_any_sleep(store, config):
    keep(store, config, "The spare key is under the blue flowerpot.")
    keep(store, config, "Leave requests go to Priya.", role="document")
    out = ask(store, config, "where is the spare key?")
    assert lines(out, "passages")[0] == "2026-10-14 user: The spare key is under the blue flowerpot."
    assert out.messages[0]["content"].startswith(f"## Memory\n{FRAMING} Today is 2026-10-15.")


def test_memory_joins_the_system_prompt_and_the_turn_stays_last(store, config):
    keep(store, config, "I live in Pune.")
    out = Recaller(store, config).build("s", [{"role": "system", "content": "Be brief."},
                                              {"role": "user", "content": "where do I live?"}], now=NOW)
    assert out.messages[0]["content"].startswith("Be brief.\n\n## Memory\n")
    assert out.messages[-1] == {"role": "user", "content": "where do I live?"}


def test_assistant_lines_and_the_current_conversation_are_not_recalled(store, config):
    keep(store, config, "Pune has lovely monsoon hills.", role="assistant")
    keep(store, config, "I love the monsoon in Pune.")
    out = ask(store, config, "monsoon plans?", history=[{"role": "user", "content": "I love the monsoon in Pune."}])
    assert lines(out, "passages") == []


def test_the_users_states_are_always_true_now(store, config, seed_state):
    seed_state(store, "s", "home_city", "Bangalore", text="I moved to Bangalore", now=NOW - 3 * DAY)
    seed_state(store, "s", "employer", "Asha's bakery", subject="Asha", text="Asha runs a bakery", now=NOW - 3 * DAY)
    out = ask(store, config, "suggest a weekend plan")
    assert lines(out, "states") == ["user · home city: Bangalore"]
    assert "Asha · employer: Asha's bakery" in lines(ask(store, config, "what does Asha do?"), "states")


def test_a_superseded_sentence_is_marked_outdated(store, config, seed_state):
    old = seed_state(store, "s", "home_city", "Pune", text="I live in Pune", now=NOW - 30 * DAY)
    store.close_state(old, NOW - 3 * DAY)
    seed_state(store, "s", "home_city", "Bangalore", text="I moved to Bangalore", now=NOW - 3 * DAY)
    out = ask(store, config, "Pune or Bangalore, where do I live?")
    assert "2026-09-15 user: I live in Pune [outdated: now user · home city: Bangalore]" in lines(out, "passages")


def test_events_are_found_and_dated(store, config):
    result = keep(store, config, "Last Saturday I went trekking to Sinhagad.")
    (line,) = store.episode_sentences([result.episode_id])
    store.add_event("s", "user went trekking to Sinhagad", NOW - 5 * DAY, True, [store.user_entity("s")], line.id)
    assert lines(ask(store, config, "when did I go trekking?"), "events") == [
        "2026-10-10: user went trekking to Sinhagad"]


def test_without_semantic_memory_only_sentences_are_recalled(store, config, seed_state):
    seed_state(store, "s", "home_city", "Bangalore", text="I moved to Bangalore", now=NOW - 3 * DAY)
    config.enable_semantic = False
    out = ask(store, config, "where do I live? Bangalore?")
    assert lines(out, "states") == [] and lines(out, "passages") == ["2026-10-12 user: I moved to Bangalore"]


def test_a_recent_memory_ranks_above_an_equally_relevant_old_one(store, config):
    keep(store, config, "My bike needs new brakes.", now=NOW - 200 * DAY)
    keep(store, config, "My bike needs a new chain.", now=NOW - DAY)
    found = Recaller(store, config).search("s", "what does my bike need?", NOW)
    assert [s.text for s, _ in found.passages][0] == "My bike needs a new chain."
    config.enable_base_level = False
    found = Recaller(store, config).search("s", "what does my bike need?", NOW)
    assert {s.text for s, _ in found.passages} == {"My bike needs new brakes.", "My bike needs a new chain."}


def test_an_unrelated_query_finds_nothing_against_the_noise_floor(noisy_store, config):
    config.novelty_warmup = 5
    for i, topic in enumerate(["trains", "mangoes", "chess", "taxes", "guitars", "monsoon", "cricket", "pottery"]):
        keep(noisy_store, config, f"Notes about {topic} number {i}.")
    found = Recaller(noisy_store, config).search("s", "quantum chromodynamics", NOW)
    assert found.passages == []


def test_budget_is_respected_and_recall_writes_nothing(store, config):
    for i in range(40):
        keep(store, config, f"Hobby number {i} is something with a long description of it.", now=NOW - i * DAY)
    before = store.stats("s")
    out = ask(store, config, "what are my hobbies?", budget=300)
    assert out.tokens <= 300 and 0 < len(out.used) and store.stats("s") == before


def test_an_empty_space_recalls_nothing(store, config):
    out = ask(store, config, "hello")
    assert out.memory_text == "" and out.messages == [{"role": "user", "content": "hello"}]
