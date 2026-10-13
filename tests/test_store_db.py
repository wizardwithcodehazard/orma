import numpy as np
import pytest

from smruti.store import index
from smruti.store.db import connect, init_schema


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_schema(c, 4)
    yield c
    c.close()


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_vector_search_is_scoped_to_space(conn):
    index.vec_put(conn, "facts", 1, "a", unit(1, 0, 0, 0))
    index.vec_put(conn, "facts", 2, "b", unit(1, 0, 0, 0))
    index.vec_put(conn, "facts", 3, "a", unit(0, 1, 0, 0))
    hits = index.vec_search(conn, "facts", "a", unit(1, 0, 0, 0), 5)
    assert [i for i, _ in hits] == [1, 3]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(0.0, abs=1e-6)


def test_vec_put_replaces_existing_row(conn):
    index.vec_put(conn, "facts", 1, "a", unit(1, 0, 0, 0))
    index.vec_put(conn, "facts", 1, "a", unit(0, 1, 0, 0))
    assert index.vec_search(conn, "facts", "a", unit(0, 1, 0, 0), 5) == [(1, pytest.approx(1.0))]


def test_text_search_ranks_matches_and_is_syntax_safe(conn):
    index.text_put(conn, "facts", 7, "a", "user lives in Pune")
    index.text_put(conn, "facts", 8, "a", "user works at Acme")
    index.text_put(conn, "facts", 9, "b", "user lives in Pune")
    hits = index.text_search(conn, "facts", "a", 'Where does the user live? "Pune"!', 5)
    assert hits[0][0] == 7
    assert 9 not in [hit.id for hit in hits]


def test_text_search_reports_term_coverage_with_the_index_stemmer(conn):
    index.text_put(conn, "facts", 1, "a", "user lives in Pune")
    index.text_put(conn, "entities", 2, "a", "New York")
    # "live" and "lives" share a porter stem, so coverage agrees with what MATCH matched.
    (hit,) = index.text_search(conn, "facts", "a", "Where does the user live?", 5)
    assert hit.query_coverage == pytest.approx(2 / 5) and hit.row_coverage == pytest.approx(2 / 4)
    (hit,) = index.text_search(conn, "entities", "a", "any new ideas?", 5)
    assert hit.row_coverage == pytest.approx(1 / 2)
    (hit,) = index.text_search(conn, "entities", "a", "I visited New York", 5)
    assert hit.row_coverage == 1.0 and hit.query_coverage == pytest.approx(2 / 4)


def test_terms_split_like_the_fts_tokenizer():
    assert index.terms("Is_allergic to, Peanuts! peanuts") == ["is", "allergic", "to", "peanuts"]


def test_text_search_with_no_terms_returns_nothing(conn):
    assert index.text_search(conn, "facts", "a", "?!", 5) == []


def test_text_drop_removes_row(conn):
    index.text_put(conn, "facts", 7, "a", "user lives in Pune")
    index.text_drop(conn, "facts", 7)
    assert index.text_search(conn, "facts", "a", "Pune", 5) == []


def test_dimension_mismatch_is_rejected(tmp_path):
    c = connect(tmp_path / "d.db")
    init_schema(c, 4)
    c.close()
    c = connect(tmp_path / "d.db")
    with pytest.raises(ValueError):
        init_schema(c, 8)
    c.close()


def test_index_functions_reject_unknown_kind(conn):
    v = unit(1, 0, 0, 0)
    with pytest.raises(ValueError):
        index.vec_put(conn, "bogus", 1, "a", v)
    with pytest.raises(ValueError):
        index.vec_drop(conn, "bogus", 1)
    with pytest.raises(ValueError):
        index.vec_search(conn, "bogus", "a", v, 1)
    with pytest.raises(ValueError):
        index.text_put(conn, "bogus", 1, "a", "x")
    with pytest.raises(ValueError):
        index.text_drop(conn, "bogus", 1)
    with pytest.raises(ValueError):
        index.text_search(conn, "bogus", "a", "x", 1)
    with pytest.raises(ValueError):
        index.text_put(conn, "predicates", 1, "a", "x")  # predicates have no text index


def test_existing_database_gains_procedure_sources_column(tmp_path):
    c = connect(tmp_path / "old.db")
    init_schema(c, 4)
    c.execute("ALTER TABLE procedures DROP COLUMN source_episode_ids")
    init_schema(c, 4)
    c.execute("INSERT INTO procedures (space, trigger, action, created_at, last_used) VALUES ('s', 't', 'a', 1, 1)")
    assert c.execute("SELECT source_episode_ids FROM procedures").fetchone()[0] == "[]"
    c.close()


def test_predicate_names_are_unique_per_space(store):
    import sqlite3

    store.add_predicate("s", "lives_in", "one")
    store.add_predicate("t", "lives_in", "one")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_predicate("s", "lives_in", "one")


def test_new_databases_have_no_candidate_facts_table(conn):
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'candidate_facts'").fetchone() is None


def test_store_stems_with_the_index_tokenizer(store):
    assert store.stems(["Peanuts allergy", ""]) == [{"peanut", "allergi"}, set()]
