import pytest

from orma import Memory, SmrutiConfig


def lives(s, place, since=""):
    return {"states": [{"s": s, "about": "user", "attribute": "home_city", "value": place, "ended": False,
                        "since": since}], "events": []}


def test_remember_recall_and_update_end_to_end(config, embedder, fake_llm):
    # sentence 1 is the user's turn, 2 the reply, 3 the next user turn
    llm = fake_llm([lives(1, "Pune"), lives(3, "Bangalore")], chat_reply="Noted!")
    mem = Memory(config=config, embedder=embedder, llm=llm)

    assert mem.chat([{"role": "user", "content": "I live in Pune"}]) == "Noted!"
    mem.sleep()
    context = mem.build_context([{"role": "user", "content": "Suggest a weekend plan"}])
    assert "user · home city: Pune" in context[0]["content"]

    mem.observe("user", "I just moved to Bangalore")
    mem.sleep()
    assert [s.text for s in mem.facts("user")] == ["user · home city: Bangalore"]
    assert len(mem.facts("user", include_closed=True)) == 2
    mem.close()


def test_anything_added_is_searchable_before_any_sleep(config, embedder, tmp_path):
    mem = Memory(config=config, embedder=embedder)
    mem.add("The wifi password at the office is on the fridge.")
    doc = tmp_path / "handbook.md"
    doc.write_text("Leave requests go to Priya.\n\nThe office closes at 7 pm.", encoding="utf-8")
    mem.add_file(doc)
    found = mem.search("who handles leave requests?")
    assert found.passages[0][0].text == "Leave requests go to Priya."
    assert found.passages[0][0].speaker == "handbook.md" and found.passages[0][0].role == "document"
    note = next(s for s, _ in mem.search("wifi password").passages)
    assert note.role == "note" and note.speaker == "user"
    mem.close()


def test_long_text_is_kept_as_several_records(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    topics = ["monsoon rains flood the old bridge", "the startup raised a seed round", "grandmother's mango pickle",
              "chess openings for beginners", "train routes across the western ghats", "fixing a leaking kitchen tap"]
    text = "\n\n".join(" ".join([f"This part is about {t}."] * 25) for t in topics)
    results = mem.add(text)
    assert len(results) > 1 and all(r.stored for r in results)
    assert mem.stats()["records"] == len(results)
    mem.close()


def test_rebuild_reads_everything_again(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([lives(1, "Pune"), lives(1, "Pune")]))
    mem.observe("user", "I live in Pune")
    mem.sleep()
    first = mem.facts()
    report = mem.rebuild()
    assert report.episodes == 1 and [s.text for s in mem.facts()] == [s.text for s in first]
    assert mem.stats()["records"] == 1  # records are kept; only what sleep built is rebuilt
    mem.close()


def test_forget_ends_a_state(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([lives(1, "Pune")]))
    mem.observe("user", "I live in Pune")
    mem.sleep()
    (state,) = mem.facts()
    assert mem.forget(state.id) is True
    assert mem.facts() == []
    assert mem.forget(state.id) is False
    mem.close()


def test_forget_is_scoped_to_the_space(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([lives(1, "Pune")]))
    mem.observe("user", "I live in Pune", space="a")
    mem.sleep("a")
    (state,) = mem.facts(space="a")
    assert mem.forget(state.id) is False  # the default space does not hold it
    assert mem.forget(state.id, space="a") is True
    mem.close()


def test_unknown_entity_has_no_facts(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    assert mem.facts("nobody") == []
    mem.close()


def test_chat_and_sleep_need_an_llm(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    with pytest.raises(RuntimeError):
        mem.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError):
        mem.sleep()
    mem.close()


def test_path_argument_sets_data_dir(tmp_path, embedder):
    mem = Memory(path=tmp_path / "mem", embedder=embedder, config=SmrutiConfig(embedding_dim=embedder.dim))
    assert (tmp_path / "mem" / "memory.db").exists()
    mem.close()


def test_timestamps_can_be_replayed_through_the_sdk(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([lives(1, "Pune")], chat_reply="ok"))
    mem.chat([{"role": "user", "content": "I live in Pune"}], now=1000.0)
    assert {e.created_at for e in mem.store.pending_episodes("default", 10)} == {1000.0}
    mem.sleep(now=2000.0)
    (state,) = mem.facts()
    assert state.valid_from == 1000.0 and state.evidence_times == [1000.0]  # when it was said, not when slept
    mem.close()
