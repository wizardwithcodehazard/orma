import pytest


def test_user_entity_is_created_once_per_space(store):
    a = store.user_entity("s")
    assert store.user_entity("s") == a
    assert store.entity(a).name == "user"
    assert store.user_entity("t") != a


def test_find_and_nearest_entities(store):
    pune = store.add_entity("s", "Pune", "city")
    assert store.find_entity("s", "pune").id == pune
    assert store.find_entity("s", "Mumbai") is None
    (entity, sim), *_ = store.nearest_entities("s", store.embed("Pune"), 1)
    assert entity.id == pune and sim == pytest.approx(1.0)


def test_add_fact_renders_and_indexes(store):
    user = store.user_entity("s")
    pune = store.add_entity("s", "Pune", "city")
    lives = store.add_predicate("s", "lives_in", "one")
    fid = store.add_fact("s", user, lives, pune, None, valid_from=10.0, source_episode_ids=[1], now=10.0)
    (fact,) = store.current_facts("s", subject_id=user)
    assert (fact.id, fact.text, fact.confidence, fact.support) == (fid, "user lives in Pune", 0.5, 1)
    assert store.text_search("facts", "s", "Pune", 5)[0][0] == fid


def test_literal_object_values(store):
    user = store.user_entity("s")
    diet = store.add_predicate("s", "follows_diet", "one")
    store.add_fact("s", user, diet, None, "vegetarian", 1.0, [1])
    assert store.current_facts("s")[0].text == "user follows diet vegetarian"


def test_reinforce_then_close_fact(store):
    user = store.user_entity("s")
    lives = store.add_predicate("s", "lives_in", "one")
    fid = store.add_fact("s", user, lives, store.add_entity("s", "Pune", "city"), None, 1.0, [1])
    store.reinforce_fact(fid, [2], now=20.0)
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and fact.confidence == pytest.approx(2 / 3)
    assert fact.source_episode_ids == [1, 2] and fact.last_used == 20.0
    store.close_fact(fid, valid_to=30.0)
    assert store.current_facts("s") == []
    (old,) = store.fact_history("s")
    assert old.valid_to == 30.0
    assert store.text_search("facts", "s", "Pune", 5) == []


def test_incident_facts_are_current_edges_in_the_space(store):
    user = store.user_entity("s")
    mom = store.add_entity("s", "mother", "person")
    pune = store.add_entity("s", "Pune", "city")
    has = store.add_predicate("s", "has_mother", "one")
    lives = store.add_predicate("s", "lives_in", "one")
    likes = store.add_predicate("s", "likes", "many")
    f1 = store.add_fact("s", user, has, mom, None, 1.0, [1])
    f2 = store.add_fact("s", mom, lives, pune, None, 1.0, [1])
    f3 = store.add_fact("s", user, likes, None, "chess", 1.0, [1])
    closed = store.add_fact("s", mom, likes, None, "tea", 1.0, [1])
    store.close_fact(closed, 5.0)
    other = store.user_entity("t")
    store.add_fact("t", other, likes, None, "go", 1.0, [1])

    assert sorted(store.incident_facts("s", [user])) == [(f1, user, mom), (f3, user, None)]
    assert sorted(store.incident_facts("s", [mom])) == [(f1, user, mom), (f2, mom, pune)]  # as object too
    assert sorted(store.incident_facts("s", [user, mom])) == [(f1, user, mom), (f2, mom, pune), (f3, user, None)]
    assert store.incident_facts("s", [other]) == []
    assert store.incident_facts("s", []) == []
    # Every edge comes back, so the fan (number of current facts) per entity is a count over the rows.
    fan = {e: sum(e in (s_, o) for _, s_, o in store.incident_facts("s", [user, mom, pune])) for e in (user, mom, pune)}
    assert fan == {user: 2, mom: 2, pune: 1}


def test_nearest_predicate(store):
    lives = store.add_predicate("s", "lives_in", "one")
    predicate, sim = store.nearest_predicate("s", store.embed("lives in"))
    assert predicate.id == lives and predicate.cardinality == "one" and sim == pytest.approx(1.0)
    assert [p.name for p in store.predicates("s")] == ["lives_in"]
