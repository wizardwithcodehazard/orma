# Smruti v1 Memory Core Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old smruti engine with the neuro-symbolic memory core from `docs/superpowers/specs/2026-10-12-local-neurosymbolic-memory-design.md`. It covers a typed SQLite graph store, novelty-gated encoding, budgeted graph+hybrid recall, LLM-extraction sleep with symbolic consolidation, and a `Memory` Python SDK.

**Architecture:** One SQLite file holds typed tables (episodes, entities, predicates, facts, summaries, procedures). Each table has a sqlite-vec index and an FTS5 index sharing its integer ids. Entities are graph nodes and facts are edges with valid-from/valid-to times. The request path (encode, recall) is pure embeddings plus SQL. Sleep calls an OpenAI-compatible LLM in JSON mode, then applies symbolic consolidation.

**Tech Stack:** Python 3.10+, SQLite (FTS5), sqlite-vec 0.1.9, fastembed (bge-small-en-v1.5, 384-d), numpy, httpx, pytest.

**Later plans:** Plan 2 covers `smruti proxy`, the CLI, MCP, agent hooks and the docs rewrite. Plan 3 covers the LongMemEval harness. Both build only on the public API defined here.

**Running tests:** use any Python ≥ 3.10 environment with `pip install -e ".[dev]"`. Commands below use `python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `smruti/__init__.py` | Public exports: `Memory`, `SmrutiConfig` |
| `smruti/config.py` | `SmrutiConfig`: every tunable and ablation switch |
| `smruti/embed.py` | `Embedder` protocol, `FastEmbedder`, vector helpers |
| `smruti/messages.py` | `message_text()` for OpenAI-style messages |
| `smruti/store/__init__.py` | Re-exports `Store`, records, `USER_ENTITY` |
| `smruti/store/db.py` | Connection, schema, vector and FTS virtual tables |
| `smruti/store/index.py` | Vector and lexical index put/drop/search helpers |
| `smruti/store/records.py` | Typed row dataclasses |
| `smruti/store/store.py` | `Store`: typed reads and writes, kept in sync with indexes |
| `smruti/encode.py` | `Encoder`: adaptive novelty gate |
| `smruti/recall/__init__.py` | `Recaller`: builds the prompt for a request |
| `smruti/recall/link.py` | Entity linking |
| `smruti/recall/rank.py` | Reciprocal-rank fusion, recency |
| `smruti/recall/pack.py` | Budgeted packing and rendering |
| `smruti/llm.py` | `LLMClient` for OpenAI-compatible endpoints |
| `smruti/sleep/__init__.py` | `Sleeper`: batching, retries, pruning |
| `smruti/sleep/extract.py` | Extraction schema, prompt, parsing |
| `smruti/sleep/consolidate.py` | `Consolidator`: symbolic consolidation |
| `smruti/memory.py` | `Memory` SDK facade |
| `tests/conftest.py` | `FakeEmbedder`, `FakeLLM`, fixtures |

---

### Task 1: Clean slate and configuration

**Files:**
- Delete: `smruti/engine/`, `smruti/framework.py`, `smruti/models.py`, `smruti/storage/`, `smruti/interfaces/`, `tests/*.py`, `benchmarks/`, `smruti_bundle.json`
- Create: `smruti/config.py` (replace), `smruti/__init__.py` (replace), `tests/test_config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Remove the old engine and its tests**

```bash
git rm -r -q smruti/engine smruti/framework.py smruti/models.py smruti/storage smruti/interfaces benchmarks smruti_bundle.json
git rm -q tests/*.py
```

- [ ] **Step 2: Replace `pyproject.toml` dependencies and metadata**

Replace the `[project]` table through `[project.scripts]` with:

```toml
[project]
name = "smruti"
version = "1.0.0.dev0"
description = "Local neuro-symbolic long-term memory for small LLMs"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
dependencies = [
    "fastembed>=0.2.0",
    "numpy>=1.20.0",
    "sqlite-vec>=0.1.6",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]
```

(The `[project.scripts]` entry is removed; Plan 2 adds the CLI back.)

- [ ] **Step 3: Write the failing test** `tests/test_config.py`

```python
from pathlib import Path

from smruti.config import SmrutiConfig


def test_db_path_lives_in_data_dir(tmp_path):
    assert SmrutiConfig(data_dir=tmp_path).db_path == tmp_path / "memory.db"


def test_data_dir_comes_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SMRUTI_DIR", str(tmp_path))
    assert SmrutiConfig().data_dir == tmp_path.resolve()


def test_data_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("SMRUTI_DIR", raising=False)
    assert SmrutiConfig().data_dir == Path.home() / ".smruti"
```

- [ ] **Step 4: Run it to verify it fails**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL (`ImportError` / `TypeError`: the old config has no `data_dir`)

- [ ] **Step 5: Write `smruti/config.py`**

```python
"""
config.py - Every tunable of the memory layer in one place, so each component can be
ablated or swept in experiments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    env_dir = os.environ.get("SMRUTI_DIR")
    return Path(env_dir).expanduser().resolve() if env_dir else Path.home() / ".smruti"


@dataclass
class SmrutiConfig:
    # Storage
    data_dir: Path = field(default_factory=default_data_dir)
    db_filename: str = "memory.db"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Encode: adaptive novelty gate
    novelty_percentile: float = 10.0
    novelty_warmup: int = 50
    novelty_window: int = 500

    # Recall
    budget_tokens: int = 4096
    chars_per_token: float = 3.5
    entity_link_k: int = 8
    entity_link_threshold: float = 0.6
    max_hops: int = 2
    hop_decay: float = 0.6
    search_k: int = 20
    recency_half_life_days: float = 14.0
    core_min_confidence: float = 0.7
    min_recent_turns: int = 2

    # Sleep
    sleep_after_episodes: int = 20
    sleep_idle_seconds: float = 120.0
    sleep_batch_episodes: int = 12
    sleep_max_attempts: int = 3
    entity_merge_threshold: float = 0.85
    predicate_merge_threshold: float = 0.8
    procedure_merge_threshold: float = 0.85
    summary_max_tokens: int = 200
    episode_retention_days: float = 7.0
    procedure_ttl_days: float = 30.0
    procedure_min_confidence: float = 0.5

    # Ablation switches
    enable_novelty_gate: bool = True
    enable_graph: bool = True
    enable_core_block: bool = True

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / self.db_filename
```

- [ ] **Step 6: Write `smruti/__init__.py`**

```python
"""smruti - local neuro-symbolic long-term memory for small LLMs."""

from smruti.config import SmrutiConfig

__version__ = "1.0.0.dev0"
__all__ = ["SmrutiConfig"]
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `python -m pytest tests/test_config.py -q`
Expected: `3 passed`

- [ ] **Step 8: Commit**

```bash
git add -A pyproject.toml smruti tests
git commit -m "refactor!: clean slate for v1 memory core; new SmrutiConfig"
```

---

### Task 2: Embeddings and test doubles

**Files:**
- Create: `smruti/embed.py`, `smruti/messages.py`, `tests/conftest.py`, `tests/test_embed.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import zlib

import numpy as np
import pytest

from smruti.config import SmrutiConfig
from smruti.embed import normalize


class FakeEmbedder:
    """Deterministic bag-of-words hashing: texts that share words get high cosine similarity."""

    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().replace("_", " ").split():
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        return normalize(out)


class FakeLLM:
    """Returns queued JSON extractions (or raises queued exceptions) and a fixed chat reply."""

    def __init__(self, extractions=(), chat_reply="ok"):
        self.extractions = list(extractions)
        self.chat_reply = chat_reply
        self.calls = []

    def chat_json(self, messages, model=None):
        self.calls.append(messages)
        result = self.extractions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        return self.chat_reply


@pytest.fixture
def config(tmp_path):
    return SmrutiConfig(data_dir=tmp_path, embedding_dim=FakeEmbedder.dim)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def store(config, embedder):
    from smruti.store import Store

    s = Store(config, embedder)
    yield s
    s.close()
```

- [ ] **Step 2: Write the failing test** `tests/test_embed.py`

```python
import numpy as np

from smruti.embed import l2_to_cosine, normalize, to_blob
from smruti.messages import message_text


def test_normalize_gives_unit_rows_and_handles_zero():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_l2_distance_converts_to_cosine_for_unit_vectors():
    assert l2_to_cosine(0.0) == 1.0
    assert abs(l2_to_cosine(float(np.sqrt(2.0)))) < 1e-9


def test_to_blob_is_float32_bytes():
    assert len(to_blob(np.ones(4))) == 16


def test_fake_embedder_reflects_word_overlap(embedder):
    a, b, c = embedder.embed(["user lives in pune", "user lives in mumbai", "quarterly tax filing"])
    assert a @ b > a @ c


def test_message_text_handles_strings_and_parts():
    assert message_text({"role": "user", "content": "hi"}) == "hi"
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]
    assert message_text({"role": "user", "content": parts}) == "a\nb"
    assert message_text({"role": "assistant", "content": None}) == ""
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_embed.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'smruti.embed'`

- [ ] **Step 4: Write `smruti/embed.py`**

```python
"""
embed.py - Text embeddings, the neural similarity primitive used for novelty, entity linking
and retrieval. Vectors are L2-normalized float32, so cosine similarity is a dot product.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized vectors."""


def normalize(vecs) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def l2_to_cosine(distance: float) -> float:
    """sqlite-vec reports L2 distance; for unit vectors cos = 1 - d^2 / 2."""
    return 1.0 - (distance * distance) / 2.0


class FastEmbedder:
    """ONNX embeddings via fastembed, loaded lazily on first use."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return normalize(np.asarray(list(self._model.embed(texts))))
```

- [ ] **Step 5: Write `smruti/messages.py`**

```python
"""messages.py - Helpers for OpenAI-style chat messages."""


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_embed.py -q`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add smruti/embed.py smruti/messages.py tests/conftest.py tests/test_embed.py
git commit -m "feat(embed): embedder protocol, fastembed backend, message helpers"
```

---

### Task 3: Schema and indexes

**Files:**
- Create: `smruti/store/db.py`, `smruti/store/index.py`, `smruti/store/__init__.py` (temporary, empty), `tests/test_store_db.py`

- [ ] **Step 1: Write the failing test** `tests/test_store_db.py`

```python
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
```

- [ ] **Step 2: Create an empty `smruti/store/__init__.py`** (filled in Task 4)

```python
"""store - typed SQLite memory graph with vector and lexical indexes."""
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_store_db.py -q`
Expected: FAIL with `ImportError: cannot import name 'index'`

- [ ] **Step 4: Write `smruti/store/db.py`**

```python
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
```

- [ ] **Step 5: Write `smruti/store/index.py`**

```python
"""
index.py - Vector (sqlite-vec) and lexical (FTS5) index operations shared by all record kinds.
Index rows use the same integer id as their base-table row.
"""

import re
import sqlite3

from smruti.embed import l2_to_cosine, to_blob


def vec_put(conn: sqlite3.Connection, kind: str, row_id: int, space: str, vec) -> None:
    conn.execute(f"DELETE FROM vec_{kind} WHERE rowid = ?", (row_id,))
    conn.execute(
        f"INSERT INTO vec_{kind} (rowid, space, embedding) VALUES (?, ?, ?)", (row_id, space, to_blob(vec))
    )


def vec_drop(conn: sqlite3.Connection, kind: str, row_id: int) -> None:
    conn.execute(f"DELETE FROM vec_{kind} WHERE rowid = ?", (row_id,))


def vec_search(conn: sqlite3.Connection, kind: str, space: str, vec, k: int) -> list[tuple[int, float]]:
    """The k nearest rows in the space as (id, cosine similarity), most similar first."""
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
    conn.execute(f"DELETE FROM fts_{kind} WHERE rowid = ?", (row_id,))
    conn.execute(f"INSERT INTO fts_{kind} (rowid, text, space) VALUES (?, ?, ?)", (row_id, text, space))


def text_drop(conn: sqlite3.Connection, kind: str, row_id: int) -> None:
    conn.execute(f"DELETE FROM fts_{kind} WHERE rowid = ?", (row_id,))


def text_search(conn: sqlite3.Connection, kind: str, space: str, text: str, k: int) -> list[tuple[int, float]]:
    """Best BM25 matches in the space as (id, score), higher is better."""
    query = fts_query(text)
    if not query:
        return []
    rows = conn.execute(
        f"SELECT rowid, bm25(fts_{kind}) AS rank FROM fts_{kind} "
        f"WHERE fts_{kind} MATCH ? AND space = ? ORDER BY rank LIMIT ?",
        (query, space, k),
    ).fetchall()
    return [(int(r[0]), -float(r[1])) for r in rows]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_store_db.py -q`
Expected: `6 passed`

- [ ] **Step 7: Commit**

```bash
git add smruti/store tests/test_store_db.py
git commit -m "feat(store): schema with sqlite-vec and FTS5 indexes"
```

---

### Task 4: Store, part 1: records, episodes and novelty stats

**Files:**
- Create: `smruti/store/records.py`, `smruti/store/store.py`, `tests/test_store_episodes.py`
- Modify: `smruti/store/__init__.py`

- [ ] **Step 1: Write the failing test** `tests/test_store_episodes.py`

```python
def add(store, text, space="s", now=100.0, role="user"):
    return store.add_episode(space, role, text, None, 1.0, store.embed(text), now=now)


def test_pending_episodes_are_ordered_and_scoped(store):
    e1 = add(store, "hello", now=100.0)
    e2 = add(store, "hi there", now=101.0, role="assistant")
    add(store, "elsewhere", space="other", now=102.0)
    assert [e.id for e in store.pending_episodes("s", 10)] == [e1, e2]
    assert store.pending_stats() == [("other", 1, 102.0), ("s", 2, 101.0)]


def test_mark_consolidated_then_prune_after_retention(store):
    eid = add(store, "hello", now=100.0)
    store.mark_consolidated([eid], now=200.0)
    assert store.pending_episodes("s", 10) == []
    assert store.prune_episodes(before=150.0) == 0
    assert store.prune_episodes(before=250.0) == 1
    assert store.episodes([eid]) == []
    assert store.vec_search("episodes", "s", store.embed("hello"), 5) == []


def test_repeated_sleep_failures_retire_episodes(store):
    eid = add(store, "hello")
    store.record_sleep_failure([eid], "boom", max_attempts=2, now=1.0)
    assert [e.id for e in store.pending_episodes("s", 10)] == [eid]
    store.record_sleep_failure([eid], "boom", max_attempts=2, now=2.0)
    assert store.pending_episodes("s", 10) == []


def test_bump_seen_counts_repeats(store):
    eid = add(store, "hello")
    store.bump_seen(eid)
    assert store.episodes([eid])[0].seen_count == 2


def test_novelty_scores_keep_a_window(store):
    for x in range(5):
        store.push_novelty("s", float(x), window=3)
    assert store.novelty_scores("s") == [2.0, 3.0, 4.0]
    assert store.novelty_scores("other") == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_store_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'Store'`

- [ ] **Step 3: Write `smruti/store/records.py`**

```python
"""records.py - Typed rows returned by the store."""

import json
from dataclasses import dataclass, fields


def from_row(cls, row):
    return cls(**{f.name: row[f.name] for f in fields(cls)})


@dataclass
class Episode:
    id: int
    space: str
    role: str
    text: str
    outcome: str | None
    novelty: float | None
    seen_count: int
    created_at: float
    consolidated_at: float | None


@dataclass
class Entity:
    id: int
    space: str
    name: str
    type: str
    mention_count: int
    last_seen: float


@dataclass
class Predicate:
    id: int
    space: str
    name: str
    cardinality: str


@dataclass
class Fact:
    id: int
    space: str
    subject_id: int
    predicate_id: int
    object_id: int | None
    object_value: str | None
    confidence: float
    support: int
    valid_from: float
    valid_to: float | None
    last_used: float
    source_episode_ids: list[int]
    subject: str
    predicate: str
    object: str

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate.replace('_', ' ')} {self.object}"

    @classmethod
    def from_row(cls, row) -> "Fact":
        values = {f.name: row[f.name] for f in fields(cls)}
        values["source_episode_ids"] = json.loads(values["source_episode_ids"])
        return cls(**values)


@dataclass
class Summary:
    id: int
    space: str
    topic: str
    text: str
    tokens: int
    updated_at: float


@dataclass
class Procedure:
    id: int
    space: str
    trigger: str
    action: str
    error_class: str | None
    successes: int
    failures: int
    created_at: float
    last_used: float

    @property
    def works(self) -> bool:
        return self.successes >= self.failures

    @property
    def confidence(self) -> float:
        """Beta-posterior mean of the majority outcome."""
        return (max(self.successes, self.failures) + 1) / (self.successes + self.failures + 2)

    @property
    def text(self) -> str:
        return f"When {self.trigger}: {self.action} ({'works' if self.works else 'fails'})"
```

- [ ] **Step 4: Write `smruti/store/store.py`** (episodes and novelty; Tasks 5–6 add the rest)

```python
"""
store.py - Typed access to the memory graph: episodes, entities, predicates, facts,
summaries and procedures, each kept in sync with its vector and lexical index.
A single re-entrant lock serializes access so the proxy's request path and background
sleep can share one store.
"""

import json
import threading
import time
from contextlib import contextmanager

import numpy as np

from smruti.config import SmrutiConfig
from smruti.embed import Embedder, FastEmbedder
from smruti.store import index
from smruti.store.db import connect, init_schema
from smruti.store.records import Episode, from_row

USER_ENTITY = "user"


def _now(now: float | None) -> float:
    return time.time() if now is None else now


def _placeholders(ids) -> str:
    return ",".join("?" * len(ids))


class Store:
    def __init__(self, config: SmrutiConfig, embedder: Embedder | None = None):
        self.config = config
        self.embedder = embedder or FastEmbedder(config.embedding_model, config.embedding_dim)
        self.lock = threading.RLock()
        self.conn = connect(config.db_path)
        init_schema(self.conn, self.embedder.dim)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    @contextmanager
    def tx(self):
        """Write transaction; nested calls join the outer transaction."""
        with self.lock:
            if self.conn.in_transaction:
                yield self.conn
                return
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def _rows(self, sql: str, params=()) -> list:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def _by_ids(self, cls, table: str, ids) -> list:
        ids = list(ids)
        if not ids:
            return []
        rows = {r["id"]: r for r in self._rows(f"SELECT * FROM {table} WHERE id IN ({_placeholders(ids)})", ids)}
        return [from_row(cls, rows[i]) for i in ids if i in rows]

    # --- Embeddings and index search -------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    def vec_search(self, kind: str, space: str, vec, k: int) -> list[tuple[int, float]]:
        with self.lock:
            return index.vec_search(self.conn, kind, space, vec, k)

    def text_search(self, kind: str, space: str, text: str, k: int) -> list[tuple[int, float]]:
        with self.lock:
            return index.text_search(self.conn, kind, space, text, k)

    # --- Episodes ----------------------------------------------------------------------

    def add_episode(self, space, role, text, outcome, novelty, vec, now=None) -> int:
        with self.tx() as c:
            episode_id = c.execute(
                "INSERT INTO episodes (space, role, text, outcome, novelty, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (space, role, text, outcome, novelty, _now(now)),
            ).lastrowid
            index.vec_put(c, "episodes", episode_id, space, vec)
            index.text_put(c, "episodes", episode_id, space, text)
        return episode_id

    def bump_seen(self, episode_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE episodes SET seen_count = seen_count + 1 WHERE id = ?", (episode_id,))

    def episodes(self, ids) -> list[Episode]:
        return self._by_ids(Episode, "episodes", ids)

    def pending_episodes(self, space: str, limit: int) -> list[Episode]:
        rows = self._rows(
            "SELECT * FROM episodes WHERE space = ? AND consolidated_at IS NULL ORDER BY created_at, id LIMIT ?",
            (space, limit),
        )
        return [from_row(Episode, r) for r in rows]

    def pending_stats(self) -> list[tuple[str, int, float]]:
        """(space, pending count, newest pending created_at) for every space with pending episodes."""
        rows = self._rows(
            "SELECT space, COUNT(*) AS pending, MAX(created_at) AS newest FROM episodes "
            "WHERE consolidated_at IS NULL GROUP BY space ORDER BY space"
        )
        return [(r["space"], r["pending"], r["newest"]) for r in rows]

    def mark_consolidated(self, ids, now: float) -> None:
        with self.tx() as c:
            c.executemany("UPDATE episodes SET consolidated_at = ? WHERE id = ?", [(now, i) for i in ids])

    def record_sleep_failure(self, ids, error: str, max_attempts: int, now: float) -> None:
        """Counts a failed sleep attempt; episodes that keep failing are retired so sleep cannot loop."""
        with self.tx() as c:
            for episode_id in ids:
                c.execute(
                    "UPDATE episodes SET sleep_attempts = sleep_attempts + 1, sleep_error = ? WHERE id = ?",
                    (error[:500], episode_id),
                )
                c.execute(
                    "UPDATE episodes SET consolidated_at = ? WHERE id = ? AND sleep_attempts >= ?",
                    (now, episode_id, max_attempts),
                )

    def prune_episodes(self, before: float) -> int:
        with self.tx() as c:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM episodes WHERE consolidated_at IS NOT NULL AND consolidated_at < ?", (before,)
                ).fetchall()
            ]
            for episode_id in ids:
                c.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
                index.vec_drop(c, "episodes", episode_id)
                index.text_drop(c, "episodes", episode_id)
        return len(ids)

    # --- Novelty statistics ------------------------------------------------------------

    def novelty_scores(self, space: str) -> list[float]:
        rows = self._rows("SELECT scores FROM novelty_stats WHERE space = ?", (space,))
        return json.loads(rows[0]["scores"]) if rows else []

    def push_novelty(self, space: str, score: float, window: int) -> None:
        with self.tx() as c:
            scores = (self.novelty_scores(space) + [float(score)])[-window:]
            c.execute(
                "INSERT INTO novelty_stats (space, scores) VALUES (?, ?) "
                "ON CONFLICT(space) DO UPDATE SET scores = excluded.scores",
                (space, json.dumps(scores)),
            )
```

- [ ] **Step 5: Replace `smruti/store/__init__.py`**

```python
"""store - typed SQLite memory graph with vector and lexical indexes."""

from smruti.store.records import Entity, Episode, Fact, Predicate, Procedure, Summary
from smruti.store.store import USER_ENTITY, Store

__all__ = ["Entity", "Episode", "Fact", "Predicate", "Procedure", "Store", "Summary", "USER_ENTITY"]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_store_episodes.py tests/test_store_db.py -q`
Expected: `11 passed`

- [ ] **Step 7: Commit**

```bash
git add smruti/store tests/test_store_episodes.py
git commit -m "feat(store): typed records, episodes and novelty statistics"
```

---

### Task 5: Store, part 2: entities, predicates, facts and graph neighborhood

**Files:**
- Modify: `smruti/store/store.py`
- Create: `tests/test_store_graph.py`

- [ ] **Step 1: Write the failing test** `tests/test_store_graph.py`

```python
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


def test_neighborhood_follows_current_edges(store):
    user = store.user_entity("s")
    mom = store.add_entity("s", "mother", "person")
    pune = store.add_entity("s", "Pune", "city")
    state = store.add_entity("s", "Maharashtra", "state")
    has = store.add_predicate("s", "has_mother", "one")
    lives = store.add_predicate("s", "lives_in", "one")
    located = store.add_predicate("s", "located_in", "one")
    f1 = store.add_fact("s", user, has, mom, None, 1.0, [1])
    f2 = store.add_fact("s", mom, lives, pune, None, 1.0, [1])
    f3 = store.add_fact("s", pune, located, state, None, 1.0, [1])
    assert store.neighborhood("s", [user], max_hops=1) == {f1: 0}
    assert store.neighborhood("s", [user], max_hops=2) == {f1: 0, f2: 1}
    assert store.neighborhood("s", [user], max_hops=3) == {f1: 0, f2: 1, f3: 2}
    store.close_fact(f2, 5.0)
    assert store.neighborhood("s", [user], max_hops=3) == {f1: 0}
    assert store.neighborhood("s", [], max_hops=3) == {}


def test_nearest_predicate(store):
    lives = store.add_predicate("s", "lives_in", "one")
    predicate, sim = store.nearest_predicate("s", store.embed("lives in"))
    assert predicate.id == lives and predicate.cardinality == "one" and sim == pytest.approx(1.0)
    assert [p.name for p in store.predicates("s")] == ["lives_in"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_store_graph.py -q`
Expected: FAIL with `AttributeError: 'Store' object has no attribute 'user_entity'`

- [ ] **Step 3: Extend `smruti/store/store.py`**

Change the records import to:

```python
from smruti.store.records import Entity, Episode, Fact, Predicate, from_row
```

Add these module-level constants below `USER_ENTITY`:

```python
_FACT_SELECT = """
SELECT f.*, s.name AS subject, p.name AS predicate, COALESCE(o.name, f.object_value) AS object
FROM facts f
JOIN entities s ON s.id = f.subject_id
JOIN predicates p ON p.id = f.predicate_id
LEFT JOIN entities o ON o.id = f.object_id
"""

# Breadth-first walk over current entity-to-entity facts; each fact gets the hop distance
# of the nearest walked entity it touches.
_NEIGHBORHOOD = """
WITH RECURSIVE walk(entity_id, hops) AS (
    SELECT value, 0 FROM json_each(?)
    UNION
    SELECT CASE WHEN f.subject_id = w.entity_id THEN f.object_id ELSE f.subject_id END, w.hops + 1
    FROM walk w JOIN facts f ON f.subject_id = w.entity_id OR f.object_id = w.entity_id
    WHERE f.space = ? AND f.valid_to IS NULL AND f.object_id IS NOT NULL AND w.hops + 1 < ?
)
SELECT f.id AS id, MIN(w.hops) AS hops
FROM walk w JOIN facts f ON f.subject_id = w.entity_id OR f.object_id = w.entity_id
WHERE f.space = ? AND f.valid_to IS NULL
GROUP BY f.id
"""
```

Append these methods to `Store`:

```python
    # --- Entities ----------------------------------------------------------------------

    def user_entity(self, space: str, now=None) -> int:
        """The entity for the person chatting; created on first use."""
        with self.lock:
            rows = self._rows("SELECT id FROM entities WHERE space = ? AND name = ?", (space, USER_ENTITY))
            return rows[0]["id"] if rows else self.add_entity(space, USER_ENTITY, "person", now=now)

    def add_entity(self, space: str, name: str, type_: str, vec=None, now=None) -> int:
        vec = self.embed(name) if vec is None else vec
        with self.tx() as c:
            entity_id = c.execute(
                "INSERT INTO entities (space, name, type, last_seen) VALUES (?, ?, ?, ?)",
                (space, name, type_, _now(now)),
            ).lastrowid
            index.vec_put(c, "entities", entity_id, space, vec)
            index.text_put(c, "entities", entity_id, space, name)
        return entity_id

    def touch_entity(self, entity_id: int, now=None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE entities SET mention_count = mention_count + 1, last_seen = ? WHERE id = ?",
                (_now(now), entity_id),
            )

    def entity(self, entity_id: int) -> Entity | None:
        found = self._by_ids(Entity, "entities", [entity_id])
        return found[0] if found else None

    def find_entity(self, space: str, name: str) -> Entity | None:
        rows = self._rows(
            "SELECT * FROM entities WHERE space = ? AND lower(name) = lower(?) ORDER BY id LIMIT 1",
            (space, name.strip()),
        )
        return from_row(Entity, rows[0]) if rows else None

    def nearest_entities(self, space: str, vec, k: int) -> list[tuple[Entity, float]]:
        hits = self.vec_search("entities", space, vec, k)
        found = {e.id: e for e in self._by_ids(Entity, "entities", [i for i, _ in hits])}
        return [(found[i], sim) for i, sim in hits if i in found]

    # --- Predicates --------------------------------------------------------------------

    def add_predicate(self, space: str, name: str, cardinality: str, vec=None) -> int:
        vec = self.embed(name.replace("_", " ")) if vec is None else vec
        with self.tx() as c:
            predicate_id = c.execute(
                "INSERT INTO predicates (space, name, cardinality) VALUES (?, ?, ?)", (space, name, cardinality)
            ).lastrowid
            index.vec_put(c, "predicates", predicate_id, space, vec)
        return predicate_id

    def predicate(self, predicate_id: int) -> Predicate | None:
        found = self._by_ids(Predicate, "predicates", [predicate_id])
        return found[0] if found else None

    def predicates(self, space: str) -> list[Predicate]:
        return [from_row(Predicate, r) for r in self._rows("SELECT * FROM predicates WHERE space = ? ORDER BY id", (space,))]

    def nearest_predicate(self, space: str, vec) -> tuple[Predicate, float] | None:
        hits = self.vec_search("predicates", space, vec, 1)
        return (self.predicate(hits[0][0]), hits[0][1]) if hits else None

    # --- Facts -------------------------------------------------------------------------

    def add_fact(self, space, subject_id, predicate_id, object_id, object_value, valid_from, source_episode_ids, now=None) -> int:
        """Inserts a current fact with one observation of support (confidence 1/2)."""
        with self.tx() as c:
            fact_id = c.execute(
                "INSERT INTO facts (space, subject_id, predicate_id, object_id, object_value, confidence, support, "
                "valid_from, last_used, source_episode_ids) VALUES (?, ?, ?, ?, ?, 0.5, 1, ?, ?, ?)",
                (space, subject_id, predicate_id, object_id, object_value, valid_from, _now(now),
                 json.dumps(sorted(set(source_episode_ids)))),
            ).lastrowid
            text = self.facts([fact_id])[0].text
            index.vec_put(c, "facts", fact_id, space, self.embed(text))
            index.text_put(c, "facts", fact_id, space, text)
        return fact_id

    def facts(self, ids) -> list[Fact]:
        ids = list(ids)
        if not ids:
            return []
        rows = {r["id"]: r for r in self._rows(_FACT_SELECT + f" WHERE f.id IN ({_placeholders(ids)})", ids)}
        return [Fact.from_row(rows[i]) for i in ids if i in rows]

    def current_facts(self, space: str, subject_id=None, predicate_id=None) -> list[Fact]:
        sql, params = _FACT_SELECT + " WHERE f.space = ? AND f.valid_to IS NULL", [space]
        if subject_id is not None:
            sql += " AND f.subject_id = ?"
            params.append(subject_id)
        if predicate_id is not None:
            sql += " AND f.predicate_id = ?"
            params.append(predicate_id)
        return [Fact.from_row(r) for r in self._rows(sql + " ORDER BY f.id", params)]

    def fact_history(self, space: str, subject_id=None) -> list[Fact]:
        """All facts including closed ones, oldest first: the timeline for 'when' questions."""
        sql, params = _FACT_SELECT + " WHERE f.space = ?", [space]
        if subject_id is not None:
            sql += " AND f.subject_id = ?"
            params.append(subject_id)
        return [Fact.from_row(r) for r in self._rows(sql + " ORDER BY f.valid_from, f.id", params)]

    def reinforce_fact(self, fact_id: int, source_episode_ids, now=None) -> None:
        """Another independent observation: support + 1, confidence = support / (support + 1)."""
        with self.tx() as c:
            row = c.execute("SELECT support, source_episode_ids FROM facts WHERE id = ?", (fact_id,)).fetchone()
            support = row["support"] + 1
            sources = sorted(set(json.loads(row["source_episode_ids"])) | set(source_episode_ids))
            c.execute(
                "UPDATE facts SET support = ?, confidence = ?, source_episode_ids = ?, last_used = ? WHERE id = ?",
                (support, support / (support + 1), json.dumps(sources), _now(now), fact_id),
            )

    def close_fact(self, fact_id: int, valid_to: float) -> None:
        """Ends a fact's validity: it stays in the timeline but leaves the search indexes."""
        with self.tx() as c:
            c.execute("UPDATE facts SET valid_to = ? WHERE id = ? AND valid_to IS NULL", (valid_to, fact_id))
            index.vec_drop(c, "facts", fact_id)
            index.text_drop(c, "facts", fact_id)

    def neighborhood(self, space: str, entity_ids, max_hops: int) -> dict[int, int]:
        """Current facts within max_hops edges of the given entities, as {fact id: hop distance}."""
        entity_ids = list(entity_ids)
        if not entity_ids or max_hops < 1:
            return {}
        rows = self._rows(_NEIGHBORHOOD, (json.dumps(entity_ids), space, max_hops, space))
        return {r["id"]: r["hops"] for r in rows}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_store_graph.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add smruti/store/store.py tests/test_store_graph.py
git commit -m "feat(store): entity-fact graph with temporal validity and neighborhood walk"
```

---

### Task 6: Store, part 3: summaries and procedures

**Files:**
- Modify: `smruti/store/store.py`
- Create: `tests/test_store_memory.py`

- [ ] **Step 1: Write the failing test** `tests/test_store_memory.py`

```python
import pytest


def test_summary_upsert_replaces_by_topic(store):
    a = store.upsert_summary("s", "cooking", "likes spicy food", 4, now=1.0)
    b = store.upsert_summary("s", "cooking", "likes spicy food, avoids peanuts", 7, now=2.0)
    assert a == b
    (summary,) = store.summaries([a])
    assert summary.text.endswith("peanuts") and summary.updated_at == 2.0
    assert store.summary("s", "cooking").id == a
    assert store.summary("s", "travel") is None
    hit = store.vec_search("summaries", "s", store.embed("cooking: likes spicy food, avoids peanuts"), 1)
    assert hit[0][0] == a


def test_procedure_evidence_and_confidence(store):
    pid = store.add_procedure("s", "npm run build fails with missing vite", "npm install vite",
                              "missing_dependency", works=True, now=1.0)
    store.add_procedure_evidence(pid, works=True, now=2.0)
    store.add_procedure_evidence(pid, works=False, now=3.0)
    (p,) = store.procedures([pid])
    assert (p.successes, p.failures, p.works, p.last_used) == (2, 1, True, 3.0)
    assert p.confidence == pytest.approx(3 / 5)
    assert p.text == "When npm run build fails with missing vite: npm install vite (works)"
    found, sim = store.nearest_procedure("s", store.embed("When npm run build fails with missing vite: npm install vite"))
    assert found.id == pid and sim == pytest.approx(1.0)


def test_prune_procedures_drops_only_stale_and_uncertain(store):
    confident = store.add_procedure("s", "a", "b", None, works=True, now=1.0)
    store.add_procedure_evidence(confident, works=True, now=1.0)       # 2-0 -> 0.75
    contested = store.add_procedure("s", "c", "d", None, works=True, now=1.0)
    store.add_procedure_evidence(contested, works=False, now=1.0)      # 1-1 -> 0.5
    recent = store.add_procedure("s", "e", "f", None, works=True, now=100.0)
    store.add_procedure_evidence(recent, works=False, now=100.0)       # 1-1 but recent
    assert store.prune_procedures(unused_since=50.0, min_confidence=0.6) == 1
    assert {p.id for p in store.procedures([confident, contested, recent])} == {confident, recent}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_store_memory.py -q`
Expected: FAIL with `AttributeError: 'Store' object has no attribute 'upsert_summary'`

- [ ] **Step 3: Extend `smruti/store/store.py`**

Change the records import to:

```python
from smruti.store.records import Entity, Episode, Fact, Predicate, Procedure, Summary, from_row
```

Append to `Store`:

```python
    # --- Summaries ---------------------------------------------------------------------

    def upsert_summary(self, space: str, topic: str, text: str, tokens: int, now=None) -> int:
        with self.tx() as c:
            c.execute(
                "INSERT INTO summaries (space, topic, text, tokens, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(space, topic) DO UPDATE SET text = excluded.text, tokens = excluded.tokens, "
                "updated_at = excluded.updated_at",
                (space, topic, text, tokens, _now(now)),
            )
            summary_id = c.execute(
                "SELECT id FROM summaries WHERE space = ? AND topic = ?", (space, topic)
            ).fetchone()["id"]
            indexed = f"{topic}: {text}"
            index.vec_put(c, "summaries", summary_id, space, self.embed(indexed))
            index.text_put(c, "summaries", summary_id, space, indexed)
        return summary_id

    def summaries(self, ids) -> list[Summary]:
        return self._by_ids(Summary, "summaries", ids)

    def summary(self, space: str, topic: str) -> Summary | None:
        rows = self._rows("SELECT * FROM summaries WHERE space = ? AND topic = ?", (space, topic))
        return from_row(Summary, rows[0]) if rows else None

    # --- Procedures --------------------------------------------------------------------

    def add_procedure(self, space, trigger, action, error_class, works: bool, vec=None, now=None) -> int:
        indexed = f"When {trigger}: {action}"
        vec = self.embed(indexed) if vec is None else vec
        now = _now(now)
        with self.tx() as c:
            procedure_id = c.execute(
                "INSERT INTO procedures (space, trigger, action, error_class, successes, failures, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (space, trigger, action, error_class, int(works), int(not works), now, now),
            ).lastrowid
            index.vec_put(c, "procedures", procedure_id, space, vec)
            index.text_put(c, "procedures", procedure_id, space, indexed)
        return procedure_id

    def add_procedure_evidence(self, procedure_id: int, works: bool, now=None) -> None:
        column = "successes" if works else "failures"
        with self.tx() as c:
            c.execute(
                f"UPDATE procedures SET {column} = {column} + 1, last_used = ? WHERE id = ?", (_now(now), procedure_id)
            )

    def procedures(self, ids) -> list[Procedure]:
        return self._by_ids(Procedure, "procedures", ids)

    def nearest_procedure(self, space: str, vec) -> tuple[Procedure, float] | None:
        hits = self.vec_search("procedures", space, vec, 1)
        return (self.procedures([hits[0][0]])[0], hits[0][1]) if hits else None

    def prune_procedures(self, unused_since: float, min_confidence: float) -> int:
        """Forgets procedures that are both stale and not confidently established."""
        stale = [from_row(Procedure, r) for r in self._rows("SELECT * FROM procedures WHERE last_used < ?", (unused_since,))]
        doomed = [p.id for p in stale if p.confidence < min_confidence]
        with self.tx() as c:
            for procedure_id in doomed:
                c.execute("DELETE FROM procedures WHERE id = ?", (procedure_id,))
                index.vec_drop(c, "procedures", procedure_id)
                index.text_drop(c, "procedures", procedure_id)
        return len(doomed)
```

- [ ] **Step 4: Run the store tests to verify they pass**

Run: `python -m pytest tests/test_store_memory.py tests/test_store_graph.py tests/test_store_episodes.py -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add smruti/store/store.py tests/test_store_memory.py
git commit -m "feat(store): topic summaries and evidence-counted procedures"
```

---

### Task 7: Encoder with adaptive novelty gate

**Files:**
- Create: `smruti/encode.py`, `tests/test_encode.py`

- [ ] **Step 1: Write the failing test** `tests/test_encode.py`

```python
import pytest

from smruti.encode import Encoder

DISTINCT = ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel", "india juliet"]


def test_first_turns_are_always_stored(store, config):
    enc = Encoder(store, config)
    first = enc.encode("s", "user", "I live in Pune")
    assert first.stored and first.novelty == pytest.approx(1.0)
    again = enc.encode("s", "user", "I live in Pune")
    assert again.stored  # still warming up
    assert again.novelty == pytest.approx(0.0, abs=1e-6)


def test_redundant_turn_is_counted_not_stored_after_warmup(store, config):
    config.novelty_warmup = 5
    enc = Encoder(store, config)
    ids = [enc.encode("s", "user", t).episode_id for t in DISTINCT]
    repeat = enc.encode("s", "user", "alpha bravo")
    assert not repeat.stored and repeat.episode_id == ids[0]
    assert store.episodes([ids[0]])[0].seen_count == 2
    assert len(store.pending_episodes("s", 100)) == 5


def test_tool_outcomes_always_stored(store, config):
    config.novelty_warmup = 5
    enc = Encoder(store, config)
    for t in DISTINCT:
        enc.encode("s", "user", t)
    assert enc.encode("s", "tool", "npm run build", outcome="failure").stored
    assert enc.encode("s", "tool", "npm run build", outcome="failure").stored


def test_gate_can_be_ablated(store, config):
    config.novelty_warmup = 5
    config.enable_novelty_gate = False
    enc = Encoder(store, config)
    for t in DISTINCT:
        enc.encode("s", "user", t)
    assert enc.encode("s", "user", "alpha bravo").stored
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_encode.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'smruti.encode'`

- [ ] **Step 3: Write `smruti/encode.py`**

```python
"""
encode.py - Per-turn encoding (no LLM). Like hippocampal gating, a turn that adds nothing
new relative to the space's own recent novelty distribution is counted, not stored.
Tool turns with outcomes are always stored: every outcome is evidence for procedures.
"""

from dataclasses import dataclass

import numpy as np

from smruti.config import SmrutiConfig
from smruti.store import Store

NOVELTY_SOURCES = ("episodes", "facts", "summaries")


@dataclass
class EncodeResult:
    stored: bool
    episode_id: int
    novelty: float


class Encoder:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def encode(self, space: str, role: str, text: str, outcome: str | None = None, now=None) -> EncodeResult:
        vec = self.store.embed(text)
        similarities = [sim for kind in NOVELTY_SOURCES for _, sim in self.store.vec_search(kind, space, vec, 1)]
        novelty = 1.0 - max(similarities, default=0.0)
        history = self.store.novelty_scores(space)
        self.store.push_novelty(space, novelty, self.config.novelty_window)

        if outcome is None and self._redundant(novelty, history):
            nearest = self.store.vec_search("episodes", space, vec, 1)
            if nearest:
                self.store.bump_seen(nearest[0][0])
                return EncodeResult(False, nearest[0][0], novelty)

        episode_id = self.store.add_episode(space, role, text, outcome, novelty, vec, now=now)
        return EncodeResult(True, episode_id, novelty)

    def _redundant(self, novelty: float, history: list[float]) -> bool:
        if not self.config.enable_novelty_gate or len(history) < self.config.novelty_warmup:
            return False
        return novelty < float(np.percentile(history, self.config.novelty_percentile))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_encode.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add smruti/encode.py tests/test_encode.py
git commit -m "feat(encode): adaptive novelty-gated episode encoding"
```

---

### Task 8: Ranking and budgeted packing

**Files:**
- Create: `smruti/recall/__init__.py` (temporary docstring only), `smruti/recall/rank.py`, `smruti/recall/pack.py`, `tests/test_rank_pack.py`

- [ ] **Step 1: Write the failing test** `tests/test_rank_pack.py`

```python
import pytest

from smruti.recall.pack import MemoryItem, count_tokens, pack
from smruti.recall.rank import recency, rrf


def test_rrf_rewards_agreement_and_normalizes():
    scores = rrf([1, 2, 3], [3, 1])
    assert scores[1] == 1.0
    assert scores[1] > scores[3] > scores[2]


def test_rrf_of_nothing_is_empty():
    assert rrf([], []) == {}


def test_recency_half_life():
    assert recency(0.0, 14 * 86400.0, 14.0) == pytest.approx(0.5)
    assert recency(100.0, 50.0, 14.0) == 1.0


def test_count_tokens_rounds_up():
    assert count_tokens("abcdefg", 3.5) == 2


def test_pack_keeps_last_turns_and_respects_budget():
    turns = [{"role": "user", "content": "x" * 70}, {"role": "assistant", "content": "y" * 70},
             {"role": "user", "content": "z" * 70}]
    items = [MemoryItem("facts", i, f"fact number {i}") for i in range(50)]
    packed = pack([], items, turns, budget=100, chars_per_token=3.5, min_recent=2)
    assert packed.turns == turns[-2:]
    assert packed.tokens <= 100
    assert 0 < len(packed.used) < 50
    assert packed.memory_text.startswith("## Memory\nRelevant facts:\n- fact number 0")


def test_pack_orders_sections_and_adds_older_turns_when_room():
    core = [MemoryItem("core", 1, "user lives in Pune")]
    ranked = [MemoryItem("summaries", 9, "travel: planning a Goa trip"),
              MemoryItem("procedures", 4, "When build fails: npm install (works)")]
    turns = [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}]
    packed = pack(core, ranked, turns, budget=1000, chars_per_token=3.5, min_recent=1)
    assert packed.turns == turns
    text = packed.memory_text
    assert text.index("Known facts about the user") < text.index("Lessons from past actions") < text.index("Background")


def test_pack_without_memory_has_no_header():
    packed = pack([], [], [{"role": "user", "content": "hi"}], 100, 3.5, 2)
    assert packed.memory_text == "" and packed.used == []
```

- [ ] **Step 2: Create `smruti/recall/__init__.py`** (replaced in Task 9)

```python
"""recall - per-request context assembly."""
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_rank_pack.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'smruti.recall.pack'`

- [ ] **Step 4: Write `smruti/recall/rank.py`**

```python
"""rank.py - Score fusion and time decay for recall."""


def rrf(*rankings: list[int], k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion of id rankings, normalized so the best item scores 1.0."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    top = max(scores.values(), default=0.0)
    return {item: score / top for item, score in scores.items()} if top else {}


def recency(last_used: float, now: float, half_life_days: float) -> float:
    """Exponential decay with the given half-life; never above 1."""
    age_days = max(0.0, now - last_used) / 86400.0
    return 0.5 ** (age_days / half_life_days)
```

- [ ] **Step 5: Write `smruti/recall/pack.py`**

```python
"""
pack.py - Budgeted prompt assembly. The newest turns are always kept; memory is added in a
fixed section order (core facts first, so they form a stable, cache-friendly prefix); older
turns fill whatever budget remains.
"""

import math
from dataclasses import dataclass, field

from smruti.messages import message_text

SECTIONS = {
    "core": "Known facts about the user",
    "facts": "Relevant facts",
    "procedures": "Lessons from past actions",
    "summaries": "Background",
}


def count_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


@dataclass
class MemoryItem:
    section: str
    id: int
    text: str


@dataclass
class Packed:
    memory_text: str
    turns: list[dict]
    used: list[MemoryItem] = field(default_factory=list)
    tokens: int = 0


def render(items: list[MemoryItem]) -> str:
    if not items:
        return ""
    lines = ["## Memory"]
    for section, title in SECTIONS.items():
        texts = [item.text for item in items if item.section == section]
        if texts:
            lines.append(f"{title}:")
            lines.extend(f"- {text}" for text in texts)
    return "\n".join(lines)


def pack(core, ranked, turns, budget: int, chars_per_token: float, min_recent: int) -> Packed:
    def cost(text: str) -> int:
        return count_tokens(text, chars_per_token)

    kept = list(turns[-min_recent:]) if min_recent > 0 else []
    older = list(turns[: len(turns) - len(kept)])
    kept_cost = sum(cost(message_text(t)) for t in kept)

    used: list[MemoryItem] = []
    for item in [*core, *ranked]:
        if cost(render(used + [item])) + kept_cost <= budget:
            used.append(item)
    memory_text = render(used)

    remaining = budget - cost(memory_text) - kept_cost
    extra: list[dict] = []
    for turn in reversed(older):
        turn_cost = cost(message_text(turn))
        if turn_cost > remaining:
            break
        extra.insert(0, turn)
        remaining -= turn_cost

    final_turns = extra + kept
    tokens = cost(memory_text) + sum(cost(message_text(t)) for t in final_turns)
    return Packed(memory_text, final_turns, used, tokens)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_rank_pack.py -q`
Expected: `7 passed`

- [ ] **Step 7: Commit**

```bash
git add smruti/recall tests/test_rank_pack.py
git commit -m "feat(recall): rank fusion, recency and budgeted packing"
```

---

### Task 9: Recaller with entity linking and graph expansion

**Files:**
- Create: `smruti/recall/link.py`, `tests/test_recall.py`
- Modify: `smruti/recall/__init__.py`

- [ ] **Step 1: Write the failing test** `tests/test_recall.py`

```python
from smruti.recall import Recaller


def build_graph(store):
    user = store.user_entity("s")
    pune = store.add_entity("s", "Pune", "city")
    mom = store.add_entity("s", "mother", "person")
    lives = store.add_predicate("s", "lives_in", "one")
    has = store.add_predicate("s", "has_mother", "one")
    return {  # same timestamp so recency cannot reorder them
        "lives": store.add_fact("s", user, lives, pune, None, 1.0, [1], now=1.0),
        "mom": store.add_fact("s", user, has, mom, None, 1.0, [1], now=1.0),
        "mom_city": store.add_fact("s", mom, lives, None, "Nagpur", 1.0, [1], now=1.0),
    }


def memory_lines(text):
    return [line for line in text.splitlines() if line.startswith("- ")]


def test_memory_is_appended_to_the_system_prompt(store, config):
    build_graph(store)
    out = Recaller(store, config).build(
        "s", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Where should I travel?"}]
    )
    assert out.messages[0]["role"] == "system"
    assert out.messages[0]["content"].startswith("Be brief.\n\n## Memory")
    assert "- user lives in Pune" in out.memory_text
    assert out.messages[-1] == {"role": "user", "content": "Where should I travel?"}


def test_graph_expansion_reaches_facts_search_misses(store, config):
    build_graph(store)
    config.search_k = 1
    with_graph = Recaller(store, config).build("s", [{"role": "user", "content": "Any gift ideas?"}])
    assert "- mother lives in Nagpur" in with_graph.memory_text
    assert len(memory_lines(with_graph.memory_text)) == 3

    config.enable_graph = False
    without = Recaller(store, config).build("s", [{"role": "user", "content": "Any gift ideas?"}])
    assert len(memory_lines(without.memory_text)) <= 1


def test_closer_facts_rank_first(store, config):
    build_graph(store)
    text = Recaller(store, config).build("s", [{"role": "user", "content": "hello"}]).memory_text
    assert text.index("user lives in Pune") < text.index("mother lives in Nagpur")


def test_core_block_holds_confident_user_facts(store, config):
    ids = build_graph(store)
    store.reinforce_fact(ids["lives"], [2])
    store.reinforce_fact(ids["lives"], [3])  # support 3 -> confidence 0.75
    text = Recaller(store, config).build("s", [{"role": "user", "content": "hello"}]).memory_text
    assert text.startswith("## Memory\nKnown facts about the user:\n- user lives in Pune")
    assert text.count("user lives in Pune") == 1


def test_budget_is_respected(store, config):
    user = store.user_entity("s")
    likes = store.add_predicate("s", "likes", "many")
    for i in range(200):
        store.add_fact("s", user, likes, None, f"hobby number {i} with a long description", 1.0, [i])
    out = Recaller(store, config).build("s", [{"role": "user", "content": "what do I like?"}], budget=300)
    assert out.tokens <= 300
    assert 0 < len(out.used) < 200


def test_recall_does_not_write(store, config):
    build_graph(store)
    before = [(f.id, f.last_used) for f in store.current_facts("s")]
    Recaller(store, config).build("s", [{"role": "user", "content": "Where do I live?"}], now=9e9)
    assert [(f.id, f.last_used) for f in store.current_facts("s")] == before
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_recall.py -q`
Expected: FAIL with `ImportError: cannot import name 'Recaller'`

- [ ] **Step 3: Write `smruti/recall/link.py`**

```python
"""
link.py - Neural entity linking: the query is mapped onto known graph entities by embedding
similarity plus exact lexical mentions. The user entity is always linked.
"""

from smruti.config import SmrutiConfig
from smruti.store import Store


def link_entities(store: Store, config: SmrutiConfig, space: str, query: str, vec) -> list[int]:
    linked = [store.user_entity(space)]
    for entity, sim in store.nearest_entities(space, vec, config.entity_link_k):
        if sim >= config.entity_link_threshold and entity.id not in linked:
            linked.append(entity.id)
    for entity_id, _ in store.text_search("entities", space, query, config.entity_link_k):
        if entity_id not in linked:
            linked.append(entity_id)
    return linked
```

- [ ] **Step 4: Replace `smruti/recall/__init__.py`**

```python
"""
recall - Per-request context assembly with no LLM calls and no writes: link the query to
graph entities, expand along current facts, fuse lexical and vector search, rank by
relevance x recency x confidence, and pack into the token budget.
"""

import time
from dataclasses import dataclass

from smruti.config import SmrutiConfig
from smruti.messages import message_text
from smruti.recall.link import link_entities
from smruti.recall.pack import MemoryItem, Packed, count_tokens, pack
from smruti.recall.rank import recency, rrf
from smruti.store import Store

SEARCHED_KINDS = ("facts", "procedures", "summaries")


@dataclass
class RecallResult:
    messages: list[dict]
    memory_text: str
    tokens: int
    used: list[MemoryItem]


class Recaller:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def build(self, space: str, messages: list[dict], budget: int | None = None, now=None) -> RecallResult:
        cfg = self.config
        now = time.time() if now is None else now
        budget = cfg.budget_tokens if budget is None else budget
        system_text = "\n\n".join(message_text(m) for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]
        query = next((message_text(m) for m in reversed(turns) if m.get("role") == "user"), "")

        core = self._core(space)
        core_ids = {item.id for item in core}
        ranked = [item for item in self._ranked(space, query, now)
                  if not (item.section == "facts" and item.id in core_ids)]
        system_cost = count_tokens(system_text, cfg.chars_per_token)
        packed = pack(core, ranked, turns, budget - system_cost, cfg.chars_per_token, cfg.min_recent_turns)
        return RecallResult(self._assemble(system_text, packed), packed.memory_text,
                            packed.tokens + system_cost, packed.used)

    def _core(self, space: str) -> list[MemoryItem]:
        if not self.config.enable_core_block:
            return []
        user_id = self.store.user_entity(space)
        facts = [f for f in self.store.current_facts(space, subject_id=user_id)
                 if f.confidence >= self.config.core_min_confidence]
        return [MemoryItem("core", f.id, f.text) for f in sorted(facts, key=lambda f: (f.predicate, f.id))]

    def _ranked(self, space: str, query: str, now: float) -> list[MemoryItem]:
        cfg = self.config
        vec = self.store.embed(query)
        fused = {
            kind: rrf([i for i, _ in self.store.vec_search(kind, space, vec, cfg.search_k)],
                      [i for i, _ in self.store.text_search(kind, space, query, cfg.search_k)])
            for kind in SEARCHED_KINDS
        }
        graph: dict[int, float] = {}
        if cfg.enable_graph:
            linked = link_entities(self.store, cfg, space, query, vec)
            graph = {fact_id: cfg.hop_decay ** hops
                     for fact_id, hops in self.store.neighborhood(space, linked, cfg.max_hops).items()}

        def fresh(timestamp: float) -> float:
            return recency(timestamp, now, cfg.recency_half_life_days)

        relevance = {fid: max(graph.get(fid, 0.0), fused["facts"].get(fid, 0.0))
                     for fid in graph.keys() | fused["facts"].keys()}
        facts = [(relevance[f.id] * fresh(f.last_used) * f.confidence, MemoryItem("facts", f.id, f.text))
                 for f in self.store.facts(sorted(relevance))]
        procedures = [(fused["procedures"][p.id] * fresh(p.last_used) * p.confidence,
                       MemoryItem("procedures", p.id, p.text))
                      for p in self.store.procedures(sorted(fused["procedures"]))]
        summaries = [(fused["summaries"][s.id] * fresh(s.updated_at), MemoryItem("summaries", s.id, f"{s.topic}: {s.text}"))
                     for s in self.store.summaries(sorted(fused["summaries"]))]

        ordered: list[MemoryItem] = []
        for group in (facts, procedures, summaries):
            ordered.extend(item for _, item in sorted(group, key=lambda pair: (-pair[0], pair[1].id)))
        return ordered

    @staticmethod
    def _assemble(system_text: str, packed: Packed) -> list[dict]:
        parts = [p for p in (system_text, packed.memory_text) if p]
        head = [{"role": "system", "content": "\n\n".join(parts)}] if parts else []
        return head + packed.turns
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_recall.py -q`
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add smruti/recall tests/test_recall.py
git commit -m "feat(recall): entity linking, graph expansion and hybrid ranking"
```

---

### Task 10: OpenAI-compatible LLM client

**Files:**
- Create: `smruti/llm.py`, `tests/test_llm.py`

- [ ] **Step 1: Write the failing test** `tests/test_llm.py`

```python
import json

import httpx
import pytest

from smruti.llm import LLMClient, LLMError


def client(handler):
    return LLMClient("http://local/v1", model="m", transport=httpx.MockTransport(handler))


def reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_chat_posts_openai_payload():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return reply("hi")

    assert client(handler).chat([{"role": "user", "content": "yo"}]) == "hi"
    assert seen["url"] == "http://local/v1/chat/completions"
    assert seen["body"]["model"] == "m" and seen["body"]["stream"] is False
    assert "response_format" not in seen["body"]


def test_chat_json_retries_once_at_temperature_zero():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return reply("not json" if len(bodies) == 1 else '{"facts": []}')

    assert client(handler).chat_json([]) == {"facts": []}
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[1]["temperature"] == 0.0


def test_chat_json_gives_up_after_retry():
    with pytest.raises(LLMError):
        client(lambda request: reply("nope")).chat_json([])


def test_http_errors_become_llm_errors():
    with pytest.raises(LLMError):
        client(lambda request: httpx.Response(500, json={"error": "boom"})).chat([])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_llm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'smruti.llm'`

- [ ] **Step 3: Write `smruti/llm.py`**

```python
"""
llm.py - Minimal client for any OpenAI-compatible chat endpoint (Ollama, llama.cpp server,
LM Studio, vLLM). Sleep uses it for JSON-mode extraction; the SDK uses it for chat.
"""

import json

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0, transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)

    def chat(self, messages: list[dict], model: str | None = None, json_mode: bool = False,
             temperature: float | None = None) -> str:
        body = {"model": model or self.model, "messages": messages, "stream": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if temperature is not None:
            body["temperature"] = temperature
        try:
            response = self.http.post("/chat/completions", json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"chat request failed: {exc}") from exc

    def chat_json(self, messages: list[dict], model: str | None = None) -> dict:
        """JSON-mode chat; retries once at temperature 0 if the output is not a JSON object."""
        for temperature in (None, 0.0):
            text = self.chat(messages, model=model, json_mode=True, temperature=temperature)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise LLMError("model did not return a JSON object")

    def close(self) -> None:
        self.http.close()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_llm.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add smruti/llm.py tests/test_llm.py
git commit -m "feat(llm): OpenAI-compatible client with JSON-mode retry"
```

---

### Task 11: Sleep extraction (neural half)

**Files:**
- Create: `smruti/sleep/__init__.py` (temporary docstring only), `smruti/sleep/extract.py`, `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** `tests/test_extract.py`

```python
from smruti.sleep.extract import ExtractionContext, build_messages, parse_extraction
from smruti.store import Episode


def episode(i, role, text, outcome=None):
    return Episode(i, "s", role, text, outcome, 1.0, 1, float(i), None)


def test_parse_keeps_valid_items_and_drops_malformed():
    x = parse_extraction({
        "entities": [{"name": "Pune", "type": "city"}, {"type": "x"}, "junk"],
        "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"},
                  {"subject": "user", "predicate": "likes"}],
        "procedures": [{"trigger": "build fails", "action": "npm install", "polarity": "fails", "error_class": "dep"}],
        "summary_updates": [{"topic": "Travel", "text": "Planning Goa"}, {"topic": "", "text": "x"}],
    })
    assert x.entities == {"Pune": "city"}
    assert [(f.subject, f.predicate, f.object, f.cardinality) for f in x.facts] == [("user", "lives_in", "Pune", "one")]
    assert x.procedures[0].works is False and x.procedures[0].error_class == "dep"
    assert x.summaries == {"travel": "Planning Goa"}


def test_parse_tolerates_missing_sections():
    x = parse_extraction({})
    assert not x.entities and not x.facts and not x.procedures and not x.summaries


def test_unknown_cardinality_defaults_to_many():
    x = parse_extraction({"facts": [{"subject": "user", "predicate": "likes", "object": "tea", "cardinality": "?"}]})
    assert x.facts[0].cardinality == "many"


def test_build_messages_includes_schema_context_and_outcomes():
    messages = build_messages(
        [episode(1, "user", "I moved to Bangalore"), episode(2, "tool", "npm run build", "failure")],
        ExtractionContext(["Pune"], ["lives_in"], {"travel": "Goa"}),
        max_words=150,
    )
    system, user = messages[0]["content"], messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert '"summary_updates"' in system and "150 words" in system
    assert "KNOWN ENTITIES: Pune" in user and "KNOWN PREDICATES: lives_in" in user
    assert "- travel: Goa" in user
    assert "[user] I moved to Bangalore" in user
    assert "[tool] [outcome: failure] npm run build" in user
```

- [ ] **Step 2: Create `smruti/sleep/__init__.py`** (replaced in Task 13)

```python
"""sleep - idle-time consolidation."""
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_extract.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'smruti.sleep.extract'`

- [ ] **Step 4: Write `smruti/sleep/extract.py`**

```python
"""
extract.py - Neural half of sleep: the local LLM reads a batch of episodes into a fixed JSON
schema. Malformed items are dropped here; meaning is resolved symbolically in consolidate.py.
"""

import json
from dataclasses import dataclass, field

SCHEMA = {
    "entities": [{"name": "string", "type": "string"}],
    "facts": [{
        "subject": "entity name",
        "predicate": "snake_case relation",
        "object": "entity name or literal value",
        "cardinality": "one|many",
        "time": "ISO date if stated, else empty",
    }],
    "procedures": [{
        "trigger": "situation",
        "action": "what was done",
        "polarity": "works|fails",
        "error_class": "short error category, or empty",
    }],
    "summary_updates": [{"topic": "short topic name", "text": "updated summary"}],
}

INSTRUCTIONS = """You turn a log of conversation turns and tool results into structured long-term memory.
Return only a JSON object with this shape:
{schema}

Guidance:
- Call the person chatting "user". Reuse names from KNOWN ENTITIES and relations from KNOWN PREDICATES when they mean the same thing.
- cardinality is "one" when the subject can hold only one value at a time (for example where someone lives now), otherwise "many".
- Record only facts that are stated or clearly implied. Do not guess.
- procedures come only from tool or action turns that have an outcome: what was tried and whether it worked.
- summary_updates: at most one per topic, at most {max_words} words, merged with EXISTING SUMMARIES on that topic.
- Use empty lists when there is nothing to record."""


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    cardinality: str
    time: str


@dataclass
class ExtractedProcedure:
    trigger: str
    action: str
    works: bool
    error_class: str


@dataclass
class Extraction:
    entities: dict[str, str] = field(default_factory=dict)  # name -> type
    facts: list[ExtractedFact] = field(default_factory=list)
    procedures: list[ExtractedProcedure] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)  # topic -> text


@dataclass
class ExtractionContext:
    entities: list[str]
    predicates: list[str]
    summaries: dict[str, str]


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _items(data: dict, key: str) -> list[dict]:
    value = data.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def parse_extraction(data: dict) -> Extraction:
    out = Extraction()
    for item in _items(data, "entities"):
        if _text(item.get("name")):
            out.entities[_text(item["name"])] = _text(item.get("type")) or "thing"
    for item in _items(data, "facts"):
        subject, predicate, obj = _text(item.get("subject")), _text(item.get("predicate")), _text(item.get("object"))
        if subject and predicate and obj:
            cardinality = "one" if _text(item.get("cardinality")).lower() == "one" else "many"
            out.facts.append(ExtractedFact(subject, predicate, obj, cardinality, _text(item.get("time"))))
    for item in _items(data, "procedures"):
        if _text(item.get("trigger")) and _text(item.get("action")):
            out.procedures.append(ExtractedProcedure(
                _text(item["trigger"]), _text(item["action"]),
                _text(item.get("polarity")).lower() != "fails", _text(item.get("error_class")),
            ))
    for item in _items(data, "summary_updates"):
        if _text(item.get("topic")) and _text(item.get("text")):
            out.summaries[_text(item["topic"]).lower()] = _text(item["text"])
    return out


def _format_episode(episode) -> str:
    outcome = f" [outcome: {episode.outcome}]" if episode.outcome else ""
    return f"[{episode.role}]{outcome} {episode.text}"


def build_messages(episodes, context: ExtractionContext, max_words: int) -> list[dict]:
    summaries = "\n".join(f"- {topic}: {text}" for topic, text in context.summaries.items()) or "(none)"
    known = (
        f"KNOWN ENTITIES: {', '.join(context.entities) or '(none)'}\n"
        f"KNOWN PREDICATES: {', '.join(context.predicates) or '(none)'}\n"
        f"EXISTING SUMMARIES:\n{summaries}"
    )
    log = "\n".join(_format_episode(e) for e in episodes)
    return [
        {"role": "system", "content": INSTRUCTIONS.format(schema=json.dumps(SCHEMA, indent=1), max_words=max_words)},
        {"role": "user", "content": f"{known}\n\nLOG:\n{log}"},
    ]


def extract(llm, model, episodes, context: ExtractionContext, max_words: int) -> Extraction:
    return parse_extraction(llm.chat_json(build_messages(episodes, context, max_words), model=model))
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_extract.py -q`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add smruti/sleep tests/test_extract.py
git commit -m "feat(sleep): schema-constrained LLM extraction"
```

---

### Task 12: Symbolic consolidation

**Files:**
- Create: `smruti/sleep/consolidate.py`, `tests/test_consolidate.py`

- [ ] **Step 1: Write the failing test** `tests/test_consolidate.py`

```python
from datetime import datetime, timezone

from smruti.sleep.consolidate import Consolidator, parse_time
from smruti.sleep.extract import parse_extraction

LIVES_PUNE = {
    "entities": [{"name": "Pune", "type": "city"}],
    "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}],
}


def apply(store, config, data, ids=(1,), now=100.0):
    return Consolidator(store, config).apply("s", parse_extraction(data), list(ids), now=now)


def test_new_fact_is_added_and_linked_to_entities(store, config):
    report = apply(store, config, LIVES_PUNE)
    assert report.facts_added == 1
    (fact,) = store.current_facts("s")
    assert fact.text == "user lives in Pune"
    assert fact.object_id == store.find_entity("s", "pune").id


def test_restated_fact_is_reinforced(store, config):
    apply(store, config, LIVES_PUNE, ids=[1])
    report = apply(store, config, LIVES_PUNE, ids=[2])
    assert report.facts_reinforced == 1
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and fact.source_episode_ids == [1, 2]


def test_contradiction_on_one_valued_relation_closes_old_fact(store, config):
    apply(store, config, LIVES_PUNE, now=100.0)
    report = apply(store, config, {
        "entities": [{"name": "Bangalore", "type": "city"}],
        "facts": [{"subject": "user", "predicate": "lives in", "object": "Bangalore", "cardinality": "one"}],
    }, now=200.0)
    assert report.facts_closed == 1
    assert [f.text for f in store.current_facts("s")] == ["user lives in Bangalore"]
    old = [f for f in store.fact_history("s") if f.object == "Pune"][0]
    assert old.valid_to == 200.0


def test_many_valued_relation_accumulates(store, config):
    for food in ("dosa", "biryani"):
        apply(store, config, {"facts": [{"subject": "user", "predicate": "likes", "object": food, "cardinality": "many"}]})
    assert sorted(f.object for f in store.current_facts("s")) == ["biryani", "dosa"]


def test_known_entity_is_reused_when_not_reextracted(store, config):
    apply(store, config, LIVES_PUNE)
    apply(store, config, {"facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]})
    (fact,) = store.current_facts("s")
    assert fact.support == 2 and fact.object_id is not None


def test_similar_entities_merge_only_when_types_match(store, config):
    config.entity_merge_threshold = 0.8
    works_at = lambda name, type_: {
        "entities": [{"name": name, "type": type_}],
        "facts": [{"subject": "user", "predicate": "works_at", "object": name, "cardinality": "one"}],
    }
    apply(store, config, works_at("Acme Corp", "company"))
    apply(store, config, works_at("Acme Corp Ltd", "Company"))
    (fact,) = store.current_facts("s")
    assert fact.object == "Acme Corp" and fact.support == 2

    apply(store, config, {"entities": [{"name": "Acme Corp Ltd", "type": "product"}]})
    assert store.find_entity("s", "Acme Corp Ltd") is not None


def test_procedures_accumulate_evidence(store, config):
    proc = {"procedures": [{"trigger": "npm run build fails with missing vite", "action": "npm install vite",
                            "polarity": "works"}]}
    apply(store, config, proc)
    report = apply(store, config, proc)
    assert report.procedures_reinforced == 1
    found, _ = store.nearest_procedure("s", store.embed("When npm run build fails with missing vite: npm install vite"))
    assert found.successes == 2


def test_summaries_are_capped(store, config):
    config.summary_max_tokens = 10
    apply(store, config, {"summary_updates": [{"topic": "Travel", "text": "word " * 100}]})
    summary = store.summary("s", "travel")
    assert len(summary.text) <= 35 and summary.tokens <= 10


def test_parse_time():
    assert parse_time("2026-01-01", 5.0) == datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    assert parse_time("", 5.0) == 5.0
    assert parse_time("someday", 5.0) == 5.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_consolidate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'smruti.sleep.consolidate'`

- [ ] **Step 3: Write `smruti/sleep/consolidate.py`**

```python
"""
consolidate.py - Symbolic half of sleep. Resolves extracted names onto the entity graph,
normalizes relations, applies temporal truth maintenance (a new value for a one-valued
relation closes the old fact rather than overwriting it), and accumulates evidence.
"""

import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone

from smruti.config import SmrutiConfig
from smruti.recall.pack import count_tokens
from smruti.sleep.extract import ExtractedFact, ExtractedProcedure, Extraction
from smruti.store import USER_ENTITY, Store


@dataclass
class ConsolidationReport:
    facts_added: int = 0
    facts_reinforced: int = 0
    facts_closed: int = 0
    procedures_added: int = 0
    procedures_reinforced: int = 0
    summaries_updated: int = 0

    def merge(self, other: "ConsolidationReport") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def parse_time(value: str, default: float) -> float:
    if not value:
        return default
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return default
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class Consolidator:
    def __init__(self, store: Store, config: SmrutiConfig):
        self.store = store
        self.config = config

    def apply(self, space: str, extraction: Extraction, episode_ids: list[int], now=None) -> ConsolidationReport:
        now = time.time() if now is None else now
        report = ConsolidationReport()
        resolved = {name.lower(): self._entity(space, name, type_, now) for name, type_ in extraction.entities.items()}
        for fact in extraction.facts:
            self._fact(space, fact, extraction.entities, resolved, episode_ids, now, report)
        for procedure in extraction.procedures:
            self._procedure(space, procedure, now, report)
        for topic, text in extraction.summaries.items():
            self._summary(space, topic, text, now)
            report.summaries_updated += 1
        return report

    def _entity(self, space: str, name: str, type_: str, now: float) -> int:
        if name.lower() == USER_ENTITY:
            return self.store.user_entity(space, now)
        exact = self.store.find_entity(space, name)
        if exact is not None:
            self.store.touch_entity(exact.id, now)
            return exact.id
        vec = self.store.embed(name)
        for entity, sim in self.store.nearest_entities(space, vec, 1):
            if sim >= self.config.entity_merge_threshold and entity.type.lower() == type_.lower():
                self.store.touch_entity(entity.id, now)
                return entity.id
        return self.store.add_entity(space, name, type_, vec=vec, now=now)

    def _predicate(self, space: str, name: str, cardinality: str) -> int:
        name = "_".join(name.lower().split())
        vec = self.store.embed(name.replace("_", " "))
        nearest = self.store.nearest_predicate(space, vec)
        if nearest is not None and nearest[1] >= self.config.predicate_merge_threshold:
            return nearest[0].id
        return self.store.add_predicate(space, name, cardinality, vec=vec)

    def _object_entity(self, space: str, name: str, resolved: dict[str, int]) -> int | None:
        if name.lower() in resolved:
            return resolved[name.lower()]
        known = self.store.find_entity(space, name)
        return known.id if known is not None else None

    def _fact(self, space, fact: ExtractedFact, types, resolved, episode_ids, now, report) -> None:
        subject_id = resolved.get(fact.subject.lower()) or self._entity(
            space, fact.subject, types.get(fact.subject, "thing"), now)
        object_id = self._object_entity(space, fact.object, resolved)
        object_value = None if object_id is not None else fact.object
        predicate_id = self._predicate(space, fact.predicate, fact.cardinality)
        current = self.store.current_facts(space, subject_id=subject_id, predicate_id=predicate_id)

        def same(existing) -> bool:
            if object_id is not None:
                return existing.object_id == object_id
            return existing.object_id is None and (existing.object_value or "").lower() == object_value.lower()

        match = next((f for f in current if same(f)), None)
        if match is not None:
            self.store.reinforce_fact(match.id, episode_ids, now)
            report.facts_reinforced += 1
            return

        valid_from = parse_time(fact.time, now)
        if self.store.predicate(predicate_id).cardinality == "one":
            for old in current:
                self.store.close_fact(old.id, max(valid_from, old.valid_from))
                report.facts_closed += 1
        self.store.add_fact(space, subject_id, predicate_id, object_id, object_value, valid_from, episode_ids, now=now)
        report.facts_added += 1

    def _procedure(self, space: str, procedure: ExtractedProcedure, now: float, report) -> None:
        vec = self.store.embed(f"When {procedure.trigger}: {procedure.action}")
        nearest = self.store.nearest_procedure(space, vec)
        if nearest is not None and nearest[1] >= self.config.procedure_merge_threshold:
            self.store.add_procedure_evidence(nearest[0].id, procedure.works, now)
            report.procedures_reinforced += 1
        else:
            self.store.add_procedure(space, procedure.trigger, procedure.action, procedure.error_class or None,
                                     procedure.works, vec=vec, now=now)
            report.procedures_added += 1

    def _summary(self, space: str, topic: str, text: str, now: float) -> None:
        cpt = self.config.chars_per_token
        text = text[: int(self.config.summary_max_tokens * cpt)].strip()
        self.store.upsert_summary(space, topic, text, count_tokens(text, cpt), now)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_consolidate.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add smruti/sleep/consolidate.py tests/test_consolidate.py
git commit -m "feat(sleep): symbolic consolidation with temporal truth maintenance"
```

---

### Task 13: Sleeper (batches, retries, pruning)

**Files:**
- Modify: `smruti/sleep/__init__.py`
- Create: `tests/test_sleep.py`

- [ ] **Step 1: Write the failing test** `tests/test_sleep.py`

```python
from smruti.encode import Encoder
from smruti.llm import LLMError
from smruti.sleep import Sleeper

PUNE = {"entities": [{"name": "Pune", "type": "city"}],
        "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]}
MOTHER = {"entities": [{"name": "mother", "type": "person"}],
          "facts": [{"subject": "user", "predicate": "has_mother", "object": "mother", "cardinality": "one"},
                    {"subject": "mother", "predicate": "lives_in", "object": "Nagpur", "cardinality": "one"}]}


def test_sleep_consolidates_in_batches(store, config, fake_llm):
    config.sleep_batch_episodes = 2
    enc = Encoder(store, config)
    for text in ["I live in Pune", "nice weather", "my mom lives in Nagpur"]:
        enc.encode("s", "user", text)
    llm = fake_llm([PUNE, MOTHER])
    report = Sleeper(store, config, llm).run("s")
    assert (report.batches, report.episodes, report.failed_batches) == (2, 3, 0)
    assert report.consolidation.facts_added == 3
    assert store.pending_episodes("s", 10) == []
    assert "KNOWN PREDICATES: lives_in" in llm.calls[1][1]["content"]


def test_failed_batch_is_retried_then_retired(store, config, fake_llm):
    config.sleep_max_attempts = 2
    Encoder(store, config).encode("s", "user", "hello")
    sleeper = Sleeper(store, config, fake_llm([LLMError("down"), LLMError("down")]))
    assert sleeper.run("s").failed_batches == 1
    assert len(store.pending_episodes("s", 10)) == 1
    assert sleeper.run("s").failed_batches == 1
    assert store.pending_episodes("s", 10) == []


def test_due_spaces_by_count_or_idle_time(store, config, fake_llm):
    config.sleep_after_episodes = 2
    config.sleep_idle_seconds = 60
    enc = Encoder(store, config)
    enc.encode("busy", "user", "one", now=1000.0)
    enc.encode("busy", "user", "two", now=1000.0)
    enc.encode("quiet", "user", "three", now=1000.0)
    enc.encode("fresh", "user", "four", now=1050.0)
    assert Sleeper(store, config, fake_llm()).due_spaces(now=1070.0) == ["busy", "quiet"]


def test_consolidated_episodes_are_pruned_after_retention(store, config, fake_llm):
    config.episode_retention_days = 1
    Encoder(store, config).encode("s", "user", "hello", now=0.0)
    Sleeper(store, config, fake_llm([{}])).run("s", now=10.0)
    report = Sleeper(store, config, fake_llm()).run("s", now=10.0 + 2 * 86400)
    assert report.pruned_episodes == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_sleep.py -q`
Expected: FAIL with `ImportError: cannot import name 'Sleeper'`

- [ ] **Step 3: Replace `smruti/sleep/__init__.py`**

```python
"""
sleep - Idle-time consolidation. Pending episodes are processed in batches: the LLM extracts
structure (neural), the consolidator integrates it into the graph (symbolic). Failed batches
are retried on later cycles and retired after sleep_max_attempts. Old raw episodes and stale,
uncertain procedures are forgotten.
"""

import time
from dataclasses import dataclass, field

from smruti.config import SmrutiConfig
from smruti.llm import LLMError
from smruti.sleep.consolidate import ConsolidationReport, Consolidator
from smruti.sleep.extract import ExtractionContext, extract
from smruti.store import Store

WORDS_PER_TOKEN = 0.75


@dataclass
class SleepReport:
    space: str
    batches: int = 0
    episodes: int = 0
    failed_batches: int = 0
    pruned_episodes: int = 0
    pruned_procedures: int = 0
    consolidation: ConsolidationReport = field(default_factory=ConsolidationReport)


class Sleeper:
    def __init__(self, store: Store, config: SmrutiConfig, llm, model: str | None = None):
        self.store = store
        self.config = config
        self.llm = llm
        self.model = model
        self.consolidator = Consolidator(store, config)

    def due_spaces(self, now=None) -> list[str]:
        """Spaces with enough pending episodes, or whose newest pending episode has gone idle."""
        now = time.time() if now is None else now
        return [space for space, pending, newest in self.store.pending_stats()
                if pending >= self.config.sleep_after_episodes or now - newest >= self.config.sleep_idle_seconds]

    def run(self, space: str, now=None) -> SleepReport:
        cfg = self.config
        now = time.time() if now is None else now
        report = SleepReport(space)
        while batch := self.store.pending_episodes(space, cfg.sleep_batch_episodes):
            ids = [e.id for e in batch]
            try:
                extraction = extract(self.llm, self.model, batch, self._context(space, batch),
                                     max_words=int(cfg.summary_max_tokens * WORDS_PER_TOKEN))
            except LLMError as exc:
                self.store.record_sleep_failure(ids, str(exc), cfg.sleep_max_attempts, now)
                report.failed_batches += 1
                break
            report.consolidation.merge(self.consolidator.apply(space, extraction, ids, now))
            self.store.mark_consolidated(ids, now)
            report.batches += 1
            report.episodes += len(ids)
        report.pruned_episodes = self.store.prune_episodes(now - cfg.episode_retention_days * 86400.0)
        report.pruned_procedures = self.store.prune_procedures(now - cfg.procedure_ttl_days * 86400.0,
                                                               cfg.procedure_min_confidence)
        return report

    def _context(self, space: str, batch) -> ExtractionContext:
        vec = self.store.embed(" ".join(e.text for e in batch))
        entities = [entity.name for entity, _ in self.store.nearest_entities(space, vec, 20)]
        predicates = [p.name for p in self.store.predicates(space)]
        summary_ids = [i for i, _ in self.store.vec_search("summaries", space, vec, 3)]
        summaries = {s.topic: s.text for s in self.store.summaries(summary_ids)}
        return ExtractionContext(entities, predicates, summaries)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_sleep.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add smruti/sleep/__init__.py tests/test_sleep.py
git commit -m "feat(sleep): batched sleep cycles with retry, retirement and forgetting"
```

---

### Task 14: `Memory` SDK and end-to-end test

**Files:**
- Create: `smruti/memory.py`, `tests/test_memory.py`
- Modify: `smruti/__init__.py`

- [ ] **Step 1: Write the failing test** `tests/test_memory.py`

```python
import pytest

from smruti import Memory, SmrutiConfig

PUNE = {"entities": [{"name": "Pune", "type": "city"}],
        "facts": [{"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one"}]}
BANGALORE = {"entities": [{"name": "Bangalore", "type": "city"}],
             "facts": [{"subject": "user", "predicate": "lives_in", "object": "Bangalore", "cardinality": "one"}]}


def test_remember_recall_and_update_end_to_end(config, embedder, fake_llm):
    llm = fake_llm([PUNE, BANGALORE], chat_reply="Noted!")
    mem = Memory(config=config, embedder=embedder, llm=llm)

    assert mem.chat([{"role": "user", "content": "I live in Pune"}]) == "Noted!"
    mem.sleep()
    context = mem.build_context([{"role": "user", "content": "Suggest a weekend plan"}])
    assert "user lives in Pune" in context[0]["content"]

    mem.observe("user", "I just moved to Bangalore")
    mem.sleep()
    assert [f.text for f in mem.facts("user")] == ["user lives in Bangalore"]
    assert len(mem.facts("user", include_closed=True)) == 2
    mem.close()


def test_forget_closes_a_fact(config, embedder, fake_llm):
    mem = Memory(config=config, embedder=embedder, llm=fake_llm([PUNE]))
    mem.observe("user", "I live in Pune")
    mem.sleep()
    (fact,) = mem.facts()
    assert mem.forget(fact.id) is True
    assert mem.facts() == []
    assert mem.forget(fact.id) is False
    mem.close()


def test_unknown_entity_has_no_facts(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    assert mem.facts("nobody") == []
    mem.close()


def test_chat_and_sleep_need_an_llm(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    with pytest.raises(RuntimeError):
        mem.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError):
        mem.sleep()
    mem.close()


def test_path_argument_sets_data_dir(tmp_path, embedder):
    mem = Memory(path=tmp_path / "mem", embedder=embedder, config=SmrutiConfig(embedding_dim=embedder.dim))
    assert (tmp_path / "mem" / "memory.db").exists()
    mem.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_memory.py -q`
Expected: FAIL with `ImportError: cannot import name 'Memory'`

- [ ] **Step 3: Write `smruti/memory.py`**

```python
"""
memory.py - Python SDK: one object wiring the store, encoder, recaller and sleeper together.

    mem = Memory(upstream="http://localhost:11434/v1", model="llama3.2:3b")
    reply = mem.chat([{"role": "user", "content": "I live in Pune"}])
    mem.sleep()
"""

import time
from pathlib import Path

from smruti.config import SmrutiConfig
from smruti.encode import EncodeResult, Encoder
from smruti.llm import LLMClient
from smruti.messages import message_text
from smruti.recall import Recaller
from smruti.sleep import SleepReport, Sleeper
from smruti.store import Fact, Store


class Memory:
    def __init__(self, path=None, upstream: str | None = None, model: str | None = None,
                 config: SmrutiConfig | None = None, embedder=None, llm=None):
        self.config = config or SmrutiConfig()
        if path is not None:
            self.config.data_dir = Path(path)
        self.store = Store(self.config, embedder)
        self.encoder = Encoder(self.store, self.config)
        self.recaller = Recaller(self.store, self.config)
        self.llm = llm or (LLMClient(upstream, model=model) if upstream else None)
        self.model = model

    def build_context(self, messages: list[dict], space: str = "default", budget: int | None = None) -> list[dict]:
        """The messages to send upstream: memory merged into the system prompt, trimmed to budget."""
        return self.recaller.build(space, messages, budget).messages

    def observe(self, role: str, text: str, space: str = "default", outcome: str | None = None) -> EncodeResult:
        return self.encoder.encode(space, role, text, outcome)

    def chat(self, messages: list[dict], space: str = "default") -> str:
        llm = self._require_llm()
        reply = llm.chat(self.build_context(messages, space), model=self.model)
        user_text = next((message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
        if user_text:
            self.observe("user", user_text, space)
        self.observe("assistant", reply, space)
        return reply

    def sleep(self, space: str = "default") -> SleepReport:
        return Sleeper(self.store, self.config, self._require_llm(), self.model).run(space)

    def facts(self, entity: str | None = None, space: str = "default", include_closed: bool = False) -> list[Fact]:
        subject_id = None
        if entity is not None:
            found = self.store.find_entity(space, entity)
            if found is None:
                return []
            subject_id = found.id
        if include_closed:
            return self.store.fact_history(space, subject_id)
        return self.store.current_facts(space, subject_id=subject_id)

    def forget(self, fact_id: int) -> bool:
        """Ends a fact's validity now; returns False if it was not current."""
        found = self.store.facts([fact_id])
        if not found or found[0].valid_to is not None:
            return False
        self.store.close_fact(fact_id, time.time())
        return True

    def close(self) -> None:
        self.store.close()
        if hasattr(self.llm, "close"):
            self.llm.close()

    def _require_llm(self):
        if self.llm is None:
            raise RuntimeError("This operation needs an LLM: pass upstream=... or llm=...")
        return self.llm
```

- [ ] **Step 4: Replace `smruti/__init__.py`**

```python
"""smruti - local neuro-symbolic long-term memory for small LLMs."""

from smruti.config import SmrutiConfig
from smruti.memory import Memory

__version__ = "1.0.0.dev0"
__all__ = ["Memory", "SmrutiConfig"]
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass (`72 passed`)

- [ ] **Step 6: Commit**

```bash
git add smruti/memory.py smruti/__init__.py tests/test_memory.py
git commit -m "feat: Memory SDK wiring encode, recall and sleep end to end"
```

---

## Self-review notes

- **Spec coverage:** §4 storage → Tasks 3–6; §5 encode → Task 7; §6 recall → Tasks 8–9 (token counting via the upstream `/tokenize` endpoint is deferred to Plan 2, where the proxy knows the upstream; the SDK uses `chars_per_token`); §7 sleep → Tasks 11–13; §9 SDK → Task 14; §10 error handling: sleep retries and retirement in Task 13, while proxy fail-open is in Plan 2; §11 tests: unit and end-to-end here, while proxy integration, eval smoke and latency tests are in Plans 2–3.
- **Ablations available:** `enable_novelty_gate`, `enable_graph`, `enable_core_block`; "no sleep" is simply not calling `sleep()`.
