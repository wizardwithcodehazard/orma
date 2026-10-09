"""
db.py - SQLite storage engine configured with Write-Ahead Logging (WAL) for sub-ms commits,
automatic schema migrations, and safe connection lifecycle management.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone

from smruti.config import smrutiConfig, get_config

logger = logging.getLogger(__name__)

# Base schema (Version 1)
SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    context TEXT,
    action TEXT NOT NULL,
    result TEXT,
    status TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_session_time ON episodes(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);

CREATE TABLE IF NOT EXISTS anti_memories (
    id TEXT PRIMARY KEY,
    signature TEXT UNIQUE NOT NULL,
    pattern TEXT NOT NULL,
    reason TEXT NOT NULL,
    suggested_fix TEXT,
    context_tags TEXT,
    times_triggered INTEGER DEFAULT 0,
    severity TEXT DEFAULT 'high',
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_anti_memories_signature ON anti_memories(signature);

CREATE TABLE IF NOT EXISTS cortical_rules (
    id TEXT PRIMARY KEY,
    rule_text TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    base_strength REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_cortical_rules_category ON cortical_rules(category);
"""

# Migrations definitions
MIGRATIONS = [
    (
        2,
        "Add consolidated_at, project_root, embeddings, and rule_edges graph table",
        [
            # Add consolidated_at to episodes
            "ALTER TABLE episodes ADD COLUMN consolidated_at REAL;",
            "CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated_at);",
            
            # Anti-memories scoping and lifecycle
            "ALTER TABLE anti_memories ADD COLUMN project_root TEXT;",
            "ALTER TABLE anti_memories ADD COLUMN is_active INTEGER DEFAULT 1;",
            "ALTER TABLE anti_memories ADD COLUMN valence TEXT DEFAULT 'inhibitory';",
            "CREATE INDEX IF NOT EXISTS idx_anti_memories_project ON anti_memories(project_root);",
            
            # Cortical rules embeddings and provenance
            "ALTER TABLE cortical_rules ADD COLUMN embedding TEXT;",
            "ALTER TABLE cortical_rules ADD COLUMN source_episode_ids TEXT;",
            "ALTER TABLE cortical_rules ADD COLUMN valence TEXT DEFAULT 'positive';",
            
            # Associative graph mesh between cortical rules
            """CREATE TABLE IF NOT EXISTS rule_edges (
                rule_id_a TEXT NOT NULL,
                rule_id_b TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (rule_id_a, rule_id_b),
                FOREIGN KEY(rule_id_a) REFERENCES cortical_rules(id) ON DELETE CASCADE,
                FOREIGN KEY(rule_id_b) REFERENCES cortical_rules(id) ON DELETE CASCADE
            );""",
            "CREATE INDEX IF NOT EXISTS idx_rule_edges_a ON rule_edges(rule_id_a);",
            "CREATE INDEX IF NOT EXISTS idx_rule_edges_b ON rule_edges(rule_id_b);",
        ]
    ),
    (
        3,
        "Add embedding to anti_memories for semantic preflight matching",
        [
            "ALTER TABLE anti_memories ADD COLUMN embedding TEXT;",
        ]
    ),
    (
        4,
        "Add expires_at TTL column to anti_memories for temporal decay",
        [
            "ALTER TABLE anti_memories ADD COLUMN expires_at REAL;",
            "CREATE INDEX IF NOT EXISTS idx_anti_memories_expires ON anti_memories(expires_at);",
        ]
    ),
    (
        5,
        "Add forgotten_audit table to track why rules or anti-memories were forgotten",
        [
            """CREATE TABLE IF NOT EXISTS forgotten_audit (
                id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                original_id TEXT,
                signature_or_text TEXT NOT NULL,
                reason TEXT NOT NULL,
                category TEXT,
                metadata_json TEXT,
                forgotten_at REAL NOT NULL
            );""",
            "CREATE INDEX IF NOT EXISTS idx_forgotten_time ON forgotten_audit(forgotten_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_forgotten_type ON forgotten_audit(item_type);",
        ]
    )
]

class DatabaseManager:
    def __init__(self, config: smrutiConfig | None = None):
        self.config = config or get_config()
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_conns = []
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensures .smruti directory, WAL mode, and migrations are applied."""
        self.config.smruti_dir.mkdir(parents=True, exist_ok=True)
        conn = self.get_connection()
        with self._lock:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.executescript(SCHEMA_V1_SQL)
            conn.commit()

            # Record v1 if schema_version is empty
            cur = conn.execute("SELECT COUNT(*) FROM schema_version WHERE version = 1")
            if cur.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                    (1, datetime.now(timezone.utc).timestamp(), "Initial base schema")
                )
                conn.commit()

            self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Executes sequential schema migrations safely."""
        existing_versions = {
            row[0] for row in conn.execute("SELECT version FROM schema_version").fetchall()
        }

        for version, desc, stmts in MIGRATIONS:
            if version not in existing_versions:
                logger.info("Applying database migration v%d: %s", version, desc)
                for stmt in stmts:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as e:
                        # Handle idempotent re-runs if column already exists
                        if "duplicate column name" in str(e).lower():
                            logger.debug("Column already exists in migration v%d: %s", version, e)
                        else:
                            raise
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                    (version, datetime.now(timezone.utc).timestamp(), desc)
                )
                conn.commit()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local persistent connection for maximum throughput."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self.config.smruti_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.config.db_path),
                timeout=10.0,
                check_same_thread=False
            )

            conn.row_factory = sqlite3.Row
            # Retry writes internally for up to 5s before raising OperationalError
            conn.execute("PRAGMA busy_timeout = 5000;")
            # Increase page cache to 8 MB (default is 2 MB)
            conn.execute("PRAGMA cache_size = -8000;")
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return self._local.conn

    def close(self) -> None:
        """Closes all open connections for this manager."""
        with self._lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug("Error closing sqlite connection: %s", e)
            self._all_conns.clear()
            if hasattr(self._local, "conn"):
                self._local.conn = None
        
        # Remove from global cache if present
        db_key = str(self.config.db_path.resolve())
        with _db_lock:
            _db_managers.pop(db_key, None)


# Thread-safe connection cache keyed by canonical database path
_default_db: DatabaseManager | None = None
_db_managers: dict[str, DatabaseManager] = {}
_db_lock = threading.Lock()

def get_db(config: smrutiConfig | None = None) -> DatabaseManager:
    """Thread-safe factory that caches DatabaseManager instances per database path."""
    global _default_db
    if _default_db is not None and config is None:
        return _default_db

    cfg = config or get_config()
    db_key = str(cfg.db_path.resolve())

    with _db_lock:
        if db_key not in _db_managers:
            mgr = DatabaseManager(cfg)
            _db_managers[db_key] = mgr
        else:
            mgr = _db_managers[db_key]
        
        if _default_db is None:
            _default_db = mgr
        return mgr

def close_all_dbs() -> None:
    """Closes and unloads all open database managers (useful for tests and shutdowns)."""
    global _default_db
    with _db_lock:
        for mgr in list(_db_managers.values()):
            mgr.close()
        _db_managers.clear()
        if _default_db is not None:
            _default_db.close()
            _default_db = None
