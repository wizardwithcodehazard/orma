import time

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


def test_run_once_survives_due_spaces_failing(config, embedder, fake_llm, monkeypatch):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm(), model="m")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mem, "due_spaces", boom)
    assert SleepScheduler(mem, poll_seconds=60).run_once() == []
    mem.close()


def test_loop_survives_exceptions(config, embedder, fake_llm, monkeypatch):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm(), model="m")
    scheduler = SleepScheduler(mem, poll_seconds=0.01)
    calls = []

    def boom(now=None):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(scheduler, "run_once", boom)
    scheduler.start()
    deadline = time.monotonic() + 5
    while len(calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(calls) >= 3
    assert scheduler.running
    scheduler.stop()
    mem.close()
