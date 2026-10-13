"""
index.py - Vector (sqlite-vec) and lexical (FTS5) index operations shared by all record kinds.
Index rows use the same integer id as their base-table row.
"""

import re
import sqlite3

from smruti.embed import l2_to_cosine, to_blob
from smruti.store.db import TEXT_KINDS, VECTOR_KINDS


def _check(kind: str, kinds: tuple) -> None:
    if kind not in kinds:
        raise ValueError(f"Unknown index kind: {kind!r}")


def vec_put(conn: sqlite3.Connection, kind: str, row_id: int, space: str, vec) -> None:
    _check(kind, VECTOR_KINDS)
    conn.execute(f"DELETE FROM vec_{kind} WHERE rowid = ?", (row_id,))
    conn.execute(
        f"INSERT INTO vec_{kind} (rowid, space, embedding) VALUES (?, ?, ?)", (row_id, space, to_blob(vec))
    )


def vec_drop(conn: sqlite3.Connection, kind: str, row_id: int) -> None:
    _check(kind, VECTOR_KINDS)
    conn.execute(f"DELETE FROM vec_{kind} WHERE rowid = ?", (row_id,))


def vec_search(conn: sqlite3.Connection, kind: str, space: str, vec, k: int) -> list[tuple[int, float]]:
    """The k nearest rows in the space as (id, cosine similarity), most similar first."""
    _check(kind, VECTOR_KINDS)
    rows = conn.execute(
        f"SELECT rowid, distance FROM vec_{kind} "
        f"WHERE embedding MATCH ? AND k = ? AND space = ? ORDER BY distance",
        (to_blob(vec), k, space),
    ).fetchall()
    return [(int(r[0]), l2_to_cosine(float(r[1]))) for r in rows]


def fts_query(text: str) -> str:
    """Tokenizes free text into an FTS5 OR-query of quoted terms, so user text can't break FTS syntax."""
    terms = dict.fromkeys(re.findall(r"\w+", text.lower()))
    return " OR ".join(f'"{term}"' for term in terms)


def text_put(conn: sqlite3.Connection, kind: str, row_id: int, space: str, text: str) -> None:
    _check(kind, TEXT_KINDS)
    conn.execute(f"DELETE FROM fts_{kind} WHERE rowid = ?", (row_id,))
    conn.execute(f"INSERT INTO fts_{kind} (rowid, text, space) VALUES (?, ?, ?)", (row_id, text, space))


def text_drop(conn: sqlite3.Connection, kind: str, row_id: int) -> None:
    _check(kind, TEXT_KINDS)
    conn.execute(f"DELETE FROM fts_{kind} WHERE rowid = ?", (row_id,))


def text_search(conn: sqlite3.Connection, kind: str, space: str, text: str, k: int) -> list[tuple[int, float]]:
    """Best BM25 matches in the space as (id, score), higher is better."""
    _check(kind, TEXT_KINDS)
    query = fts_query(text)
    if not query:
        return []
    rows = conn.execute(
        f"SELECT rowid, bm25(fts_{kind}) AS rank FROM fts_{kind} "
        f"WHERE fts_{kind} MATCH ? AND space = ? ORDER BY rank LIMIT ?",
        (query, space, k),
    ).fetchall()
    return [(int(r[0]), -float(r[1])) for r in rows]
