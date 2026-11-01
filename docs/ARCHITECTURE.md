# Orma Architecture

An engineering overview of the **Orma** long-term cognitive memory engine for small local LLMs and AI coding assistants.

---

## Core Principle

> **Episodes are the memory; states are the index of what is true now.**

Orma borrows the organizational architecture of human memory and adapts it to consumer laptop hardware (Apple Silicon, 8–16 GB RAM, local GPUs):

| Cognitive Concept | Laptop Reality | Orma Implementation |
|---|---|---|
| **Episodic Memory:** Stored instantly; facts consolidate slowly during sleep | Storage is cheap and lossless; the local LLM is the scarce resource | Every turn, document, or note is appended in full immediately (< 5ms). Sleep spends one short, asynchronous LLM call per batch only on structure |
| **Semantic Memory:** Represents what is currently true | Single-valued state is easy to maintain deterministically in code | Only **states** (single value at a time, superseded on change) and **dated events** are structured. Everything else is retrieved as exact cited sentences |
| **Associative & Cue-Driven:** Favors recency, repetition, and relevance | Full-text and vector search indexes are cheap and fast | Hybrid vector (sqlite-vec) + BM25 (FTS5) fusion, weighted against an adaptive space noise floor, with bounded ACT-R recency decay |
| **Non-Monotonic Logic:** Defeasible exceptions don't corrupt general rules | Naive memory systems destructively overwrite baseline habits | Schema-driven conditional exceptions (`item.condition`) are preserved alongside baseline truths without false overwrites |
| **Inhibitory Control:** Negative boundaries suppress repeated mistakes | Standard autoregressive attention focuses on forbidden tokens | Prohibited rules are isolated, attributed via schema, and hoisted to the top of prompts (`[FORBIDDEN: ...]`) with negative framing semantics |
| **Reconsolidation:** Memory is updated as understanding deepens | Data can be re-indexed by a stronger reader | `orma rebuild` re-processes every kept record with an updated or stronger model |

---

## High-Level Architecture

Orma provides two primary interfaces into the same local cognitive SQLite engine:

```
+-----------------------------------------------------------------------------------------+
|                                    CLIENT INTERFACES                                    |
|   1. OpenAI-Compatible Proxy (orma proxy / orma up) -> Web UIs, LangChain, OpenAI SDK   |
|   2. Model Context Protocol Server (orma mcp) -> Claude Code, Cursor, Windsurf, IDEs   |
|   3. Native Python SDK (from orma import Memory) -> Custom agents, local apps           |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                         ONLINE RETRIEVAL ENGINE (< 25ms, 0 LLM Calls)                   |
|   - Multi-Channel Search: FTS5 BM25 + sqlite-vec (bge-small-en-v1.5)                    |
|   - Reciprocal Rank Fusion (RRF) calibrated against dynamic noise floor                 |
|   - Spreading Activation (ACT-R): Associative entity traversal with Anderson's Fan      |
|   - Stochastic Ranker: Hawkes temporal burst + Ebbinghaus spaced stability              |
|   - Budget Packer: Inhibitory hoisting ([FORBIDDEN: ...]) + Sutra bracket packing       |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                    OFFLINE ASYNCHRONOUS SLEEP CONSOLIDATION (Background Worker)         |
|   - Batcher: Gathers pending episodes when idle (120s) or threshold (20 episodes)       |
|   - Sleep Model: JSON schema-constrained extraction (states, events, conditions)        |
|   - Verification Gate: Line citations, grounding check, calendar time anchoring        |
|   - Defeasible & Inhibitory Gate: Namespaces forbidden_* rules; isolates exceptions     |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                         DURABLE STORAGE ENGINE (100% Local SQLite)                      |
|   ~/.orma/memory.db (WAL mode, nested transactions, FTS5 indexes, sqlite-vec)           |
+-----------------------------------------------------------------------------------------+
```

---

## Subsystems

### 1. Text & Structural Normalization (`orma.text`)
- **Sentence Splitting:** Accurate sentence division at line breaks and punctuation boundaries, guarding against abbreviations ("Dr.", "e.g.", "v0.1").
- **Code Block Isolation:** Fenced code blocks are preserved intact or collapsed to token markers during recall packing to prevent code from polluting linguistic search.
- **Word Stemming & Paraphrase Windows:** Normalizes terms for BM25 search and phrase overlap calculations.

### 2. Episodic Encoding (`orma.encode`)
- **Zero-Latency Ingestion:** Ingests turns in `< 5ms` with immediate embedding via `bge-small-en-v1.5` (FastEmbed ONNX runtime).
- **Novelty Gating:** Sentences with cosine similarity `>= 0.97` against existing memories increment `seen_count` rather than duplicating storage.
- **Role Isolation:** Distinguishes provenance (`user`, `assistant`, `tool`, `note`, `document`). Only human/tool facts are eligible for line citations.

### 3. Recall & Cognitive Ranking (`orma.recall`)
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF) fuses BM25 text rank with vector distance.
- **Dynamic Noise Floor:** Automatically calculates the 50th-percentile similarity baseline of the space; scores below the floor contribute zero weight.
- **Spreading Activation (ACT-R):** Traverses the entity link graph. Concepts with lower fan (specific entities like `"Dr. Rao"`) transmit strong associative activation; broad hubs (`"Pune"`) dissipate energy.
- **Cognitive Decay Suite:**
  - *ACT-R Base-Level Decay:* Recency and frequency power-law scaling.
  - *Hawkes Self-Excitation:* 30-minute conversational burst half-life.
  - *Ebbinghaus Spaced Repetition:* Repeated spaced citations dynamically scale memory half-life stability.
- **Sutra Dense Packing & Inhibitory Hoisting:**
  - Formats memories into compact bracket notation: `[user.role: systems engineer @ Zerodha]`.
  - Negative rules (`forbidden_*`) are hoisted to the **very top** of the system prompt as `[FORBIDDEN: <rule>]` accompanied by dynamic negative framing instructions.

### 4. Sleep Consolidation (`orma.sleep`)
- **Non-Blocking Background Worker:** Runs when the machine is idle for 120 seconds or after 20 episodes.
- **Constrained Schema Extraction (`extract.py`):** Uses structured JSON decoding for `states` (`about`, `attribute`, `value`, `ended`, `since`, `condition`, `prohibited`) and `events` (`what`, `when`, `who`).
- **Defeasible Epistemic Engine (`consolidate.py`):** Purely schema-driven condition inspection:
  - Categorical assertions (`condition: ""`) supersede outdated contradictory states.
  - Conditional assertions (`condition: "sore throat"`) are preserved alongside baseline habits without closing them.
  - Prohibited rules (`prohibited: true`) are namespaced as `forbidden_<attribute>` for inhibitory control.

### 5. Serving Interfaces (`orma.serve`)
- **MCP Server (`mcp.py`):** Speaks Model Context Protocol over standard `stdio JSON-RPC 2.0`, exposing `orma_recall`, `orma_remember`, `orma_forbid`, `orma_facts`, and `orma_forget`.
- **OpenAI-Compatible Proxy (`proxy.py`):** Intercepts `/v1/chat/completions`, injects recalled memory into the system prompt, streams upstream replies byte-for-byte, and non-blockingly appends completed turns.
- **CLI & Web UI (`cli.py`):** Typer-based CLI for all administrative commands (`orma recall`, `orma add`, `orma search`, `orma up`, `orma mcp`).

---

## Data Model

| SQLite Table | Role & Content |
|---|---|
| `episodes` | Full, lossless record log: raw text, role, speaker, source, timestamp. Never pruned. |
| `sentences` | Sentences parsed from each record, indexed for hybrid search and cited by sleep. |
| `entities` | Named subjects and objects (`user`, people, projects, places). |
| `entity_aliases` | Cross-referenced user nicknames and aliases mapped back to `user`. |
| `attributes` | Normalized attribute names (`home_city`, `forbidden_powershell_chain`). |
| `states` | Single-valued assertions: subject, attribute, value, `valid_from`, `valid_to`, line citations, evidence times. |
| `events` | Dated occurrences, participant references, and occurrence timestamps. |
| `vec_*` / `fts_*` | Virtual tables for `sqlite-vec` vector embeddings and FTS5 BM25 text search. |

---

## Configuration & Ablation Switches

All switches are accessible via `OrmaConfig` in Python or configuration flags:

```python
from orma import Memory, OrmaConfig

config = OrmaConfig(
    enable_semantic=True,             # Sleep consolidation (states & events)
    enable_novelty_gate=True,         # Drop duplicates (cosine >= 0.97)
    enable_similarity_weighting=True, # Space noise floor calibration
    enable_base_level=True,           # ACT-R power-law recency decay
    enable_hawkes=True,               # Hawkes conversational temporal bursts
    enable_ebbinghaus=True,           # Ebbinghaus spaced stability scaling
    enable_spreading_activation=True, # Associative entity graph traversal
    enable_defeasible_logic=True,     # Non-monotonic conditional exception logic
    enable_sutra_packing=False,       # Dense bracket packing (-40% prompt tokens)
    budget_tokens=4096,               # Token ceiling for memory prompt blocks
)
mem = Memory(config=config)
```
