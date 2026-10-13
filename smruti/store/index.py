"""
index.py - Vector (sqlite-vec) and lexical (FTS5) index operations shared by all record kinds.
Index rows use the same integer id as their base-table row.
"""

import re
import sqlite3
from typing import NamedTuple

from smruti.embed import l2_to_cosine, to_blob
from smruti.store.db import FTS_TOKENIZER, TEXT_KINDS, VECTOR_KINDS


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


def terms(text: str) -> list[str]:
    """Distinct lowercased alphanumeric tokens in order, split like FTS5's unicode61 tokenizer splits
    (underscores and punctuation separate tokens)."""
    return list(dict.fromkeys(re.findall(r"[^\W_]+", text.lower())))


def fts_query(text: str) -> str:
    """Tokenizes free text into an FTS5 OR-query of quoted terms, so user text can't break FTS syntax."""
    return " OR ".join(f'"{term}"' for term in terms(text))


def stems(conn: sqlite3.Connection, texts: list[str]) -> list[set[str]]:
    """The set of index terms of each text, produced by the FTS tables' own tokenizer (porter stemming
    included) via a per-connection TEMP scratch table and its fts5vocab view, so symbolic term
    comparisons agree exactly with what MATCH matched. Only TEMP state is written."""
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.stem_scratch USING fts5(text, tokenize = '{FTS_TOKENIZER}')")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.stem_terms USING fts5vocab(temp, stem_scratch, instance)")
    conn.execute("DELETE FROM temp.stem_scratch")
    conn.executemany("INSERT INTO temp.stem_scratch (rowid, text) VALUES (?, ?)", enumerate(texts))
    out: list[set[str]] = [set() for _ in texts]
    for doc, term in conn.execute("SELECT doc, term FROM temp.stem_terms"):
        out[int(doc)].add(term)
    conn.execute("DELETE FROM temp.stem_scratch")
    return out


class TextHit(NamedTuple):
    id: int
    query_coverage: float  # fraction of the query's distinct terms that the row contains
    row_coverage: float  # fraction of the row's distinct terms that the query contains


def text_put(conn: sqlite3.Connection, kind: str, row_id: int, space: str, text: str) -> None:
    _check(kind, TEXT_KINDS)
    conn.execute(f"DELETE FROM fts_{kind} WHERE rowid = ?", (row_id,))
    conn.execute(f"INSERT INTO fts_{kind} (rowid, text, space) VALUES (?, ?, ?)", (row_id, text, space))


def text_drop(conn: sqlite3.Connection, kind: str, row_id: int) -> None:
    _check(kind, TEXT_KINDS)
    conn.execute(f"DELETE FROM fts_{kind} WHERE rowid = ?", (row_id,))


def text_search(conn: sqlite3.Connection, kind: str, space: str, text: str, k: int) -> list[TextHit]:
    """Best BM25 matches in the space, best first, with their term overlap with the query in both
    directions (BM25 ranks; overlap says how much of the query a hit actually matched)."""
    _check(kind, TEXT_KINDS)
    query = fts_query(text)
    if not query:
        return []
    rows = conn.execute(
        f"SELECT rowid, text FROM fts_{kind} "
        f"WHERE fts_{kind} MATCH ? AND space = ? ORDER BY bm25(fts_{kind}) LIMIT ?",
        (query, space, k),
    ).fetchall()
    if not rows:
        return []
    query_terms, *row_terms = stems(conn, [text] + [r[1] for r in rows])
    return [TextHit(int(r[0]), len(query_terms & t) / len(query_terms) if query_terms else 0.0,
                    len(query_terms & t) / len(t) if t else 0.0)
            for r, t in zip(rows, row_terms)]
