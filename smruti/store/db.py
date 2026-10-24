"""
db.py - SQLite connection and schema: typed tables for the memory plus sqlite-vec (vector) and FTS5
(lexical) indexes keyed by the same integer ids.

Episodes are kept records (a chat turn, a note, a document part, a tool result), never deleted; their
sentences are what recall finds and what sleep cites. States (one value at a time, superseded when it
changes) and events (dated happenings) are built from sentences in sleep and always point at the
sentence that states them.
"""

import sqlite3
from pathlib import Path

import sqlite_vec

VECTOR_KINDS = ("episodes", "sentences", "entities", "attributes", "states", "events")
TEXT_KINDS = ("sentences", "entities", "states", "events")
SCHEMA_VERSION = "2"

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
    sleep_error TEXT,
    source TEXT NOT NULL DEFAULT 'chat',
    speaker TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_episodes_pending ON episodes(space, consolidated_at, created_at);

CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sentences_episode ON sentences(episode_id, position);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    last_seen REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(space, name);

CREATE TABLE IF NOT EXISTS entity_aliases (
    space TEXT NOT NULL,
    alias TEXT NOT NULL,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (space, alias)
);

CREATE TABLE IF NOT EXISTS attributes (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    name TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_attributes_name ON attributes(space, name);

CREATE TABLE IF NOT EXISTS states (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    attribute_id INTEGER NOT NULL REFERENCES attributes(id),
    value TEXT NOT NULL,
    valid_from REAL NOT NULL,
    valid_from_stated INTEGER NOT NULL DEFAULT 0,
    valid_to REAL,
    sentence_id INTEGER NOT NULL REFERENCES sentences(id),
    ended_by INTEGER REFERENCES sentences(id),
    evidence TEXT NOT NULL DEFAULT '[]',
    evidence_times TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_states_subject ON states(space, subject_id, attribute_id, valid_to);
CREATE INDEX IF NOT EXISTS idx_states_sentence ON states(sentence_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    gist TEXT NOT NULL,
    occurred_at REAL NOT NULL,
    occurred_stated INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    sentence_id INTEGER NOT NULL REFERENCES sentences(id)
);
CREATE INDEX IF NOT EXISTS idx_events_sentence ON events(sentence_id);

CREATE TABLE IF NOT EXISTS event_participants (
    event_id INTEGER NOT NULL REFERENCES events(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (event_id, entity_id)
);

CREATE TABLE IF NOT EXISTS sleep_leases (space TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at REAL NOT NULL);

CREATE TABLE IF NOT EXISTS novelty_stats (space TEXT PRIMARY KEY, scores TEXT NOT NULL);
"""

# Shared by every FTS table and by index.stems, so symbolic term matching uses the index's own tokenizer.
FTS_TOKENIZER = "porter unicode61"

# Columns added to tables that existed in version 1 databases: (table, column, declaration).
MIGRATIONS = (
    ("episodes", "source", "TEXT NOT NULL DEFAULT 'chat'"),
    ("episodes", "speaker", "TEXT NOT NULL DEFAULT ''"),
)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection, dim: int) -> None:
    """Creates or upgrades the schema. A version 1 database keeps its episodes (and its old, now unused, tables);
    `smruti rebuild` splits them into sentences and consolidates them again."""
    conn.executescript(SCHEMA)
    for table, column, declaration in MIGRATIONS:
        if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('embedding_dim', ?)", (str(dim),))
    elif int(row["value"]) != dim:
        raise ValueError(f"Database was created with embedding_dim={row['value']}, config has {dim}")
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
    for kind in VECTOR_KINDS:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{kind} USING vec0("
            f"space text partition key, embedding float[{dim}])"
        )
    for kind in TEXT_KINDS:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS fts_{kind} USING fts5("
            f"text, space UNINDEXED, tokenize = '{FTS_TOKENIZER}')"
        )
