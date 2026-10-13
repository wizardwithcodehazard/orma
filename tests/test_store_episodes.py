def add(store, text, space="s", now=100.0, role="user"):
    return store.add_episode(space, role, text, None, 1.0, store.embed(text), now=now)


def test_pending_episodes_are_ordered_and_scoped(store):
    e1 = add(store, "hello", now=100.0)
    e2 = add(store, "hi there", now=101.0, role="assistant")
    add(store, "elsewhere", space="other", now=102.0)
    assert [e.id for e in store.pending_episodes("s", 10)] == [e1, e2]
    assert store.pending_stats() == [("other", 1, 102.0), ("s", 2, 101.0)]


def test_mark_consolidated_then_prune_after_retention(store):
    eid = add(store, "hello", now=100.0)
    store.mark_consolidated([eid], now=200.0)
    assert store.pending_episodes("s", 10) == []
    assert store.prune_episodes(before=150.0) == 0
    assert store.prune_episodes(before=250.0) == 1
    assert store.episodes([eid]) == []
    assert store.vec_search("episodes", "s", store.embed("hello"), 5) == []


def test_repeated_sleep_failures_retire_episodes(store):
    eid = add(store, "hello")
    store.record_sleep_failure([eid], "boom", max_attempts=2, now=1.0)
    assert [e.id for e in store.pending_episodes("s", 10)] == [eid]
    store.record_sleep_failure([eid], "boom", max_attempts=2, now=2.0)
    assert store.pending_episodes("s", 10) == []


def test_bump_seen_counts_repeats(store):
    eid = add(store, "hello")
    store.bump_seen(eid)
    assert store.episodes([eid])[0].seen_count == 2


def test_novelty_scores_keep_a_window(store):
    for x in range(5):
        store.push_novelty("s", float(x), window=3)
    assert store.novelty_scores("s") == [2.0, 3.0, 4.0]
    assert store.novelty_scores("other") == []
