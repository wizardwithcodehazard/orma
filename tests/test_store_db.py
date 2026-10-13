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
    assert 9 not in [i for i, _ in hits]


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
