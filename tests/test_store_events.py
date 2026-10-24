def sentence(store, text, space="s"):
    episode_id = store.add_episode(space, "user", text, None, 1.0, store.embed(text), speaker="user")
    (sentence_id,) = store.add_sentences(space, store.episodes([episode_id])[0], [text], [store.embed(text)])
    return sentence_id


def test_event_round_trip_and_participants(store):
    user = store.user_entity("s")
    asha = store.add_entity("s", "Asha", "person")
    goa_line, cake_line = sentence(store, "We went to Goa"), sentence(store, "I baked a cake")
    goa = store.add_event("s", "user went to Goa with Asha", 100.0, True, [user, asha, asha], goa_line, now=200.0)
    cake = store.add_event("s", "user baked a cake", 150.0, False, [user], cake_line, now=200.0)
    store.add_event("other", "user baked bread", 150.0, False, [user], sentence(store, "bread", "other"))

    (event,) = store.events([goa])
    assert (event.space, event.gist, event.occurred_at, event.occurred_stated, event.created_at,
            event.sentence_id) == ("s", "user went to Goa with Asha", 100.0, True, 200.0, goa_line)
    assert sorted(store.event_participants(goa)) == sorted([user, asha])
    assert [e.id for e in store.events_citing(goa_line)] == [goa]
    assert [e.id for e in store.space_events("s")] == [goa, cake]
    assert [i for i, _ in store.vec_search("events", "s", store.embed("baked a cake"), 1)] == [cake]
    assert [hit.id for hit in store.text_search("events", "s", "Goa", 5)] == [goa]
