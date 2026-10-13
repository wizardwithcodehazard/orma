import pytest

from smruti import Memory, SmrutiConfig

PUNE = {"entities": [{"name": "Pune", "type": "city"}],
        "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]}
BANGALORE = {"entities": [{"name": "Bangalore", "type": "city"}],
             "facts": [{"subject": "user", "predicate": "lives_in", "object": "Bangalore", "cardinality": "one"}]}


def test_remember_recall_and_update_end_to_end(config, embedder, fake_llm):
    llm = fake_llm([PUNE, BANGALORE], chat_reply="Noted!")
    mem = Memory(config=config, embedder=embedder, llm=llm)

    assert mem.chat([{"role": "user", "content": "I live in Pune"}]) == "Noted!"
    mem.sleep()
    context = mem.build_context([{"role": "user", "content": "Suggest a weekend plan"}])
    assert "user lives in Pune" in context[0]["content"]

    mem.observe("user", "I just moved to Bangalore")
    mem.sleep()
    assert [f.text for f in mem.facts("user")] == ["user lives in Bangalore"]
    assert len(mem.facts("user", include_closed=True)) == 2
    mem.close()


def test_forget_closes_a_fact(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([PUNE]))
    mem.observe("user", "I live in Pune")
    mem.sleep()
    (fact,) = mem.facts()
    assert mem.forget(fact.id) is True
    assert mem.facts() == []
    assert mem.forget(fact.id) is False
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


def test_forget_is_scoped_to_the_space(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([PUNE]))
    mem.observe("user", "I live in Pune", space="a")
    mem.sleep("a")
    (fact,) = mem.facts(space="a")
    assert mem.forget(fact.id) is False  # default space does not hold it
    assert len(mem.facts(space="a")) == 1
    assert mem.forget(fact.id, space="a") is True
    mem.close()


def test_timestamps_can_be_replayed_through_the_sdk(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([PUNE], chat_reply="ok"))
    mem.chat([{"role": "user", "content": "I live in Pune"}], now=1000.0)
    assert {e.created_at for e in mem.store.pending_episodes("default", 10)} == {1000.0}
    mem.sleep(now=2000.0)
    (fact,) = mem.facts()
    assert fact.last_used == 1000.0  # evidence time, not sleep time
    mem.observe("user", "I like tea", now=3000.0)
    assert mem.store.pending_episodes("default", 10)[0].created_at == 3000.0
    old = mem.build_context([{"role": "user", "content": "where do I live"}], now=1000.0 + 14 * 86400)
    fresh = mem.build_context([{"role": "user", "content": "where do I live"}], now=1000.0)
    assert "user lives in Pune" in old[0]["content"] and "user lives in Pune" in fresh[0]["content"]
    mem.close()
