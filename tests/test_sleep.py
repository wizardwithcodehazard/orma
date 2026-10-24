from smruti.encode import Encoder
from smruti.sleep import Sleeper
from smruti.sleep.extract import JSON_SCHEMA

NOTHING = {"states": [], "events": []}


def turn(store, config, text, role="user", now=None):
    return Encoder(store, config).encode("s", role, text, now=now, speaker=role).episode_id


def test_sleep_reads_records_in_batches(store, config, fake_llm):
    config.sleep_batch_episodes = 2
    for i in range(3):
        turn(store, config, f"fact number {i} about trains", now=float(i))
    llm = fake_llm([NOTHING, NOTHING])
    report = Sleeper(store, config, llm, "m").run("s", now=100.0)
    assert (report.batches, report.episodes, report.failed_batches) == (2, 3, 0)
    assert store.pending_episodes("s", 10) == [] and llm.json_kwargs[0] == {"schema": JSON_SCHEMA, "temperature": 0.0}


def test_lines_are_numbered_and_assistant_turns_are_context(store, config, fake_llm):
    user = turn(store, config, "I live in Pune.", now=1.0)
    turn(store, config, "Pune is lovely! You probably love the hills.", role="assistant", now=2.0)
    llm = fake_llm([NOTHING])
    Sleeper(store, config, llm, "m").run("s", now=100.0)
    prompt = llm.calls[0][1]["content"]
    (line,) = store.episode_sentences([user])
    assert f"[s{line.id} 1970-01-01 user] I live in Pune." in prompt
    assert "[context 1970-01-01 assistant] Pune is lovely!" in prompt


def test_a_batch_with_nothing_citable_needs_no_model_call(store, config, fake_llm):
    turn(store, config, "Anything else I can help with?", role="assistant")
    llm = fake_llm([])
    report = Sleeper(store, config, llm, "m").run("s")
    assert report.episodes == 1 and llm.calls == [] and store.pending_episodes("s", 10) == []


def test_without_semantic_memory_sleep_only_marks_records_read(store, config, fake_llm):
    config.enable_semantic = False
    turn(store, config, "I live in Pune.")
    llm = fake_llm([])
    assert Sleeper(store, config, llm, "m").run("s").episodes == 1 and llm.calls == []


def test_failed_batch_is_retried_then_retired(store, config, fake_llm):
    config.sleep_max_attempts = 2
    turn(store, config, "I live in Pune.")
    llm = fake_llm([RuntimeError("model down"), RuntimeError("model down")])
    assert Sleeper(store, config, llm, "m").run("s").failed_batches == 1
    assert len(store.pending_episodes("s", 10)) == 1
    Sleeper(store, config, llm, "m").run("s")
    assert store.pending_episodes("s", 10) == []  # retired after the cap


def test_an_off_schema_answer_counts_as_a_failed_attempt(store, config, fake_llm):
    turn(store, config, "I live in Pune.")
    report = Sleeper(store, config, fake_llm([{"facts": []}]), "m").run("s")
    assert report.failed_batches == 1 and len(store.pending_episodes("s", 10)) == 1


def test_due_spaces_by_count_or_idle_time(store, config, fake_llm):
    config.sleep_after_episodes, config.sleep_idle_seconds = 2, 50.0
    turn(store, config, "first thing", now=100.0)
    sleeper = Sleeper(store, config, fake_llm(), "m")
    assert sleeper.due_spaces(now=120.0) == [] and sleeper.due_spaces(now=151.0) == ["s"]
    turn(store, config, "second thing", now=121.0)
    assert sleeper.due_spaces(now=122.0) == ["s"]


def test_second_sleeper_does_nothing_while_the_lease_is_held(store, config, fake_llm):
    turn(store, config, "I live in Pune.")
    store.acquire_sleep_lease("s", "someone-else", 600.0, now=0.0)
    report = Sleeper(store, config, fake_llm([]), "m").run("s", now=10.0)
    assert report.batches == 0 and len(store.pending_episodes("s", 10)) == 1


def test_batches_respect_the_prompt_budget(store, config, fake_llm):
    config.sleep_prompt_tokens = 30
    for i in range(3):
        turn(store, config, f"Sentence number {i} is about the weather in the hills.", now=float(i))
    llm = fake_llm([NOTHING, NOTHING, NOTHING])
    assert Sleeper(store, config, llm, "m").run("s").batches == 3


def test_an_oversized_record_is_cut_to_the_budget(store, config, fake_llm):
    config.sleep_prompt_tokens = 40
    turn(store, config, " ".join(f"Sentence {i} is long enough to count." for i in range(20)))
    llm = fake_llm([NOTHING])
    Sleeper(store, config, llm, "m").run("s")
    assert llm.calls[0][1]["content"].count("[s") < 20


def test_context_shows_known_names_aliases_attributes_and_states(store, config, fake_llm, seed_state):
    seed_state(store, "s", "home_city", "Pune", text="I live in Pune")
    store.add_alias("s", "Sahil", store.user_entity("s"))
    turn(store, config, "Thinking about moving house soon.")
    llm = fake_llm([NOTHING])
    Sleeper(store, config, llm, "m").run("s", now=100.0)
    prompt = llm.calls[0][1]["content"]
    assert "user (also called: sahil)" in prompt and "home_city" in prompt and "- user · home city: Pune" in prompt
