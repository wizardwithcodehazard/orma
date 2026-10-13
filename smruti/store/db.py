"""
db.py - SQLite connection and schema: typed tables for the memory graph plus sqlite-vec
(vector) and FTS5 (lexical) indexes keyed by the same integer ids.
"""

import sqlite3
from pathlib import Path

import sqlite_vec

VECTOR_KINDS = ("episodes", "entities", "predicates", "facts", "summaries", "procedures", "candidates")
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

CREATE TABLE IF NOT EXISTS entity_aliases (
    space TEXT NOT NULL,
    alias TEXT NOT NULL,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (space, alias)
);

CREATE TABLE IF NOT EXISTS predicates (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    name TEXT NOT NULL,
    cardinality TEXT NOT NULL CHECK (cardinality IN ('one', 'many'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_predicates_name ON predicates(space, name);

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
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    valid_from_stated INTEGER NOT NULL DEFAULT 0
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
    last_used REAL NOT NULL,
    source_episode_ids TEXT NOT NULL DEFAULT '[]'
);

-- Facts only one extraction pass produced: held back from the graph (and from recall) until a later,
-- independent batch repeats them, or dropped after candidate_ttl_days.
CREATE TABLE IF NOT EXISTS candidate_facts (
    id INTEGER PRIMARY KEY,
    space TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    cardinality TEXT NOT NULL,
    time TEXT NOT NULL,
    source_episode_ids TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sleep_leases (space TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at REAL NOT NULL);

CREATE TABLE IF NOT EXISTS novelty_stats (space TEXT PRIMARY KEY, scores TEXT NOT NULL);
"""


# Shared by every FTS table and by index.stems, so symbolic term matching uses the index's own tokenizer.
FTS_TOKENIZER = "porter unicode61"


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
    conn.executescript(SCHEMA)
    if "source_episode_ids" not in {r["name"] for r in conn.execute("PRAGMA table_info(procedures)")}:
        conn.execute("ALTER TABLE procedures ADD COLUMN source_episode_ids TEXT NOT NULL DEFAULT '[]'")
    if "valid_from_stated" not in {r["name"] for r in conn.execute("PRAGMA table_info(facts)")}:
        conn.execute("ALTER TABLE facts ADD COLUMN valid_from_stated INTEGER NOT NULL DEFAULT 0")
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
            f"text, space UNINDEXED, tokenize = '{FTS_TOKENIZER}')"
        )
