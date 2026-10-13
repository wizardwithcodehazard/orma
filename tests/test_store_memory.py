import pytest


def test_summary_upsert_replaces_by_topic(store):
    a = store.upsert_summary("s", "cooking", "likes spicy food", 4, now=1.0)
    b = store.upsert_summary("s", "cooking", "likes spicy food, avoids peanuts", 7, now=2.0)
    assert a == b
    (summary,) = store.summaries([a])
    assert summary.text.endswith("peanuts") and summary.updated_at == 2.0
    assert store.summary("s", "cooking").id == a
    assert store.summary("s", "travel") is None
    hit = store.vec_search("summaries", "s", store.embed("cooking: likes spicy food, avoids peanuts"), 1)
    assert hit[0][0] == a


def test_procedure_evidence_and_confidence(store):
    pid = store.add_procedure("s", "npm run build fails with missing vite", "npm install vite",
                              "missing_dependency", works=True, now=1.0)
    store.add_procedure_evidence(pid, works=True, now=2.0)
    store.add_procedure_evidence(pid, works=False, now=3.0)
    (p,) = store.procedures([pid])
    assert (p.successes, p.failures, p.works, p.last_used) == (2, 1, True, 3.0)
    assert p.confidence == pytest.approx(3 / 5)
    assert p.text == "When npm run build fails with missing vite: npm install vite (works)"
    found, sim = store.nearest_procedure("s", store.embed("When npm run build fails with missing vite: npm install vite"))
    assert found.id == pid and sim == pytest.approx(1.0)


def test_prune_procedures_drops_only_stale_and_uncertain(store):
    confident = store.add_procedure("s", "a", "b", None, works=True, now=1.0)
    store.add_procedure_evidence(confident, works=True, now=1.0)       # 2-0 -> 0.75
    contested = store.add_procedure("s", "c", "d", None, works=True, now=1.0)
    store.add_procedure_evidence(contested, works=False, now=1.0)      # 1-1 -> 0.5
    recent = store.add_procedure("s", "e", "f", None, works=True, now=100.0)
    store.add_procedure_evidence(recent, works=False, now=100.0)       # 1-1 but recent
    assert store.prune_procedures(unused_since=50.0, min_confidence=0.6) == 1
    assert {p.id for p in store.procedures([confident, contested, recent])} == {confident, recent}


def test_prune_procedures_clears_vector_and_text_indexes(store):
    pid = store.add_procedure("s", "disk full", "clean cache", None, works=False, now=1.0)
    assert store.text_search("procedures", "s", "disk full", 5)
    assert store.prune_procedures(unused_since=50.0, min_confidence=0.7) == 1
    assert store.vec_search("procedures", "s", store.embed("When disk full: clean cache"), 5) == []
    assert store.text_search("procedures", "s", "disk full", 5) == []
    assert store.procedures([pid]) == []


def test_nearest_procedure_is_none_after_prune(store):
    store.add_procedure("s", "disk full", "clean cache", None, works=False, now=1.0)
    store.prune_procedures(unused_since=50.0, min_confidence=0.7)
    assert store.nearest_procedure("s", store.embed("When disk full: clean cache")) is None


def test_sleep_lease_is_exclusive_until_released_or_expired(store):
    assert store.acquire_sleep_lease("s", "a", ttl=10, now=100.0)
    assert store.acquire_sleep_lease("s", "a", ttl=10, now=105.0)  # same holder renews
    assert not store.acquire_sleep_lease("s", "b", ttl=10, now=110.0)
    assert store.acquire_sleep_lease("other", "b", ttl=10, now=110.0)  # leases are per space
    assert store.acquire_sleep_lease("s", "b", ttl=10, now=115.0)  # expired
    store.release_sleep_lease("s", "a")  # not the holder: no effect
    assert not store.acquire_sleep_lease("s", "a", ttl=10, now=116.0)
    store.release_sleep_lease("s", "b")
    assert store.acquire_sleep_lease("s", "a", ttl=10, now=116.0)


def test_add_fact_and_procedure_accept_precomputed_vectors(store):
    user = store.user_entity("s")
    pred = store.add_predicate("s", "likes", "many")
    calls = []
    real = store.embed
    store.embed = lambda text: (calls.append(text), real(text))[1]
    store.add_fact("s", user, pred, None, "tea", 1.0, [1], vec=real("user likes tea"))
    store.add_procedure("s", "a", "b", None, works=True, vec=real("When a: b"), source_episode_ids=[4])
    assert calls == []
    assert store.vec_search("facts", "s", real("user likes tea"), 1)[0][1] > 0.99


def test_user_entity_does_not_embed_while_holding_the_lock(store):
    held = []
    real = store.embed
    store.embed = lambda text: (held.append(store.lock._is_owned()), real(text))[1]
    first = store.user_entity("s")
    assert held == [False]
    assert store.user_entity("s") == first and held == [False]
