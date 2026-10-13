"""
db.py - SQLite connection and schema: typed tables for the memory graph plus sqlite-vec
(vector) and FTS5 (lexical) indexes keyed by the same integer ids.
"""

import sqlite3
from pathlib import Path

import sqlite_vec

VECTOR_KINDS = ("episodes", "entities", "predicates", "facts", "summaries", "procedures")
TEXT_KINDS = ("episodes", "entities", "facts", "summaries", "procedures")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    outcome TEXT,
    novelty REAL,
    seen_count INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    consolidated_at REAL,
    sleep_attempts INTEGER NOT NULL DEFAULT 0,
    sleep_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_pending ON episodes(space, consolidated_at, created_at);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    last_seen REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(space, name);

CREATE TABLE IF NOT EXISTS predicates (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    name TEXT NOT NULL,
    cardinality TEXT NOT NULL CHECK (cardinality IN ('one', 'many'))
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    predicate_id INTEGER NOT NULL REFERENCES predicates(id),
    object_id INTEGER REFERENCES entities(id),
    object_value TEXT,
    confidence REAL NOT NULL,
    support INTEGER NOT NULL,
    valid_from REAL NOT NULL,
    valid_to REAL,
    last_used REAL NOT NULL,
    source_episode_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(space, subject_id, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(space, object_id, valid_to);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    topic TEXT NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (space, topic)
);

CREATE TABLE IF NOT EXISTS procedures (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    trigger TEXT NOT NULL,
    action TEXT NOT NULL,
    error_class TEXT,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS novelty_stats (space TEXT PRIMARY KEY, scores TEXT NOT NULL);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection, dim: int) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('embedding_dim', ?)", (str(dim),))
    elif int(row["value"]) != dim:
        raise ValueError(f"Database was created with embedding_dim={row['value']}, config has {dim}")
    for kind in VECTOR_KINDS:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{kind} USING vec0("
            f"space text partition key, embedding float[{dim}])"
        )
    for kind in TEXT_KINDS:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS fts_{kind} USING fts5("
            f"text, space UNINDEXED, tokenize = 'porter unicode61')"
        )
