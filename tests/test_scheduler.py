from smruti import Memory
from smruti.serve.scheduler import SleepScheduler


def test_run_once_sleeps_due_spaces(config, embedder, fake_llm):
    config.sleep_after_episodes = 1
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([{"facts": []}]), model="m")
    mem.observe("user", "hello", space="s")
    assert SleepScheduler(mem, poll_seconds=60).run_once() == ["s"]
    assert mem.store.pending_episodes("s", 10) == []
    mem.close()


def test_run_once_waits_until_a_model_is_known(config, embedder, fake_llm):
    config.sleep_after_episodes = 1
    mem = Memory(config=config, embedder=embedder, llm=fake_llm())
    mem.observe("user", "hello", space="s")
    assert SleepScheduler(mem, poll_seconds=60).run_once() == []
    mem.close()


def test_run_once_never_raises(config, embedder, fake_llm, monkeypatch):
    config.sleep_after_episodes = 1
    mem = Memory(config=config, embedder=embedder, llm=fake_llm(), model="m")
    mem.observe("user", "hello", space="s")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mem, "sleep", boom)
    assert SleepScheduler(mem, poll_seconds=60).run_once() == []
    mem.close()


def test_start_and_stop(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm(), model="m")
    scheduler = SleepScheduler(mem, poll_seconds=0.01)
    scheduler.start()
    scheduler.stop()
    assert not scheduler.running
    mem.close()
