def test_event_round_trip_and_participant_lookup(store):
    user = store.user_entity("s")
    asha = store.add_entity("s", "Asha", "person")
    goa = store.add_event("s", "user went to Goa with Asha", 100.0, True, [user, asha, asha], [2, 1], now=200.0)
    cake = store.add_event("s", "user baked a cake", 150.0, False, [user], [3], now=200.0)
    store.add_event("other", "user baked bread", 150.0, False, [user], [4])

    (event,) = store.events([goa])
    assert (event.space, event.gist, event.occurred_at, event.occurred_stated, event.created_at,
            event.source_episode_ids) == ("s", "user went to Goa with Asha", 100.0, True, 200.0, [1, 2])
    assert sorted(store.event_participants(goa)) == sorted([user, asha])
    assert sorted(store.events_for_entities("s", [user, asha])) == sorted([(goa, user), (goa, asha), (cake, user)])
    assert store.events_for_entities("s", [asha]) == [(goa, asha)]
    assert [e.id for e in store.batch_events("s", [1, 2])] == [goa]
    assert [i for i, _ in store.vec_search("events", "s", store.embed("baked a cake"), 1)] == [cake]
    assert [hit.id for hit in store.text_search("events", "s", "Goa", 5)] == [goa]
