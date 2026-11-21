# Orma Architecture & Mathematical Specifications

An in-depth technical treatise on the **Orma** long-term cognitive memory engine, designed for consumer laptop hardware (Apple Silicon, 8–16 GB RAM, local GPUs) and local open-weight language models (3B to 8B parameters).

---

## 1. Foundational Epistemology: The Tripartite Sweet Spot

Orma bridges three foundational disciplines to build a memory engine that is mathematically sound, linguistically robust, and hardware-efficient:

```
+---------------------------------------------------------------------------------------------------+
|                                  THE TRIPARTITE ARCHITECTURAL ENSEMBLE                            |
+-----------------------------------+-----------------------------------+---------------------------+
| 1. Cognitive Neuroscience         | 2. Paninian Formal Grammar        | 3. Computational Math     |
| Behavioral dynamics, mnemonics,   | Structural linguistic tracking,   | Hardware-grounded metric  |
| and biological forgetting curves  | pronoun scope, and sutra encoding | space IR on local SQLite  |
+-----------------------------------+-----------------------------------+---------------------------+
| - Method of Loci (Memory Palace)  | - Anuvṛtti (Pronoun Inheritance)  | - Rocchio PRF Centroids   |
| - ACT-R Base-Level Decay          | - Sūtra Bracket Compression       | - Personalized PageRank   |
| - Hawkes Temporal Self-Excitation | - Negative Inhibitory Hoisting    | - FlashRank ONNX TinyBERT |
| - Ebbinghaus Spaced Stability     | - Defeasible Condition Isolation  | - Maximal Marginal Rel.   |
+-----------------------------------+-----------------------------------+---------------------------+
```

### Core Axiom
> **Episodes are the memory; states are just an index of what is true right now.**

Human memory does not invoke a neural network to retrieve stored records. Storage is lossless, append-only, and persistent. Structured states are derived indices that evolve over time without destroying episodic evidence.

---

## 2. High-Level System Architecture

Orma decouples sub-65ms online zero-LLM retrieval from offline, non-blocking background sleep consolidation:

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
|                     ONLINE ZERO-LLM RETRIEVAL ENGINE (< 65ms on CPU)                    |
|   - Method of Loci: Cosine similarity search against dialogue session checkpoints       |
|   - Hybrid First Pass: FTS5 BM25 + sqlite-vec dense embeddings via RRF fusion           |
|   - Stage 1: Rocchio Pseudo-Relevance Feedback (vector centroid query expansion)        |
|   - Stage 2: Personalized PageRank (PPR power iteration on bipartite entity graph)      |
|   - Stage 3: FlashRank ONNX TinyBERT cross-encoder (full query-passage cross-attention) |
|   - Stage 4: Maximal Marginal Relevance (MMR diversity packing to eliminate duplicates) |
|   - Cognitive Activation: ACT-R recency decay + Hawkes conversational temporal burst    |
|   - Deterministic Guard: Top-of-prompt inhibitory hoisting ([FORBIDDEN: ...])           |
|   - Context Packing: Dense Sutra format ([user.attr: val]) yielding 414 tokens (98.4%)  |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                    OFFLINE ASYNCHRONOUS SLEEP CONSOLIDATION (Background Worker)         |
|   - Idle Trigger: Fires after 120s of conversational silence or 20 pending episodes     |
|   - Local Model: Constrained JSON schema extraction via Ollama / llama.cpp / vLLM       |
|   - Deterministic Verification Gate: Line citations check, calendar time anchoring      |
|   - Defeasible & Inhibitory Gate: Namespaces forbidden_* rules; isolates exceptions     |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                         DURABLE STORAGE ENGINE (100% Local SQLite)                      |
|   ~/.orma/memory.db (WAL mode, nested transactions, FTS5 indexes, sqlite-vec virtuals)  |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Retrieval Specifications

Orma's online retrieval operates entirely on CPU without invoking language models, guaranteeing sub-65ms latency and zero hallucination risk:

### 3.1 Method of Loci (Topological Memory Palace)
Conversation dialogues are pegged to topological room checkpoints. Each locus stores an embedding vector:
```
Score(L_k, q) = (v_q . v_Lk) / (||v_q|| * ||v_Lk||)
```
Top-scoring loci identify relevant historical dialogue chapters, constraining search to semantically relevant conversation sessions.

### 3.2 Reciprocal Rank Fusion & Dynamic Noise Floor
First-pass candidates are retrieved across dense vector search (`sqlite-vec`) and keyword search (`FTS5 BM25`):
```
RRF_Score(d) = (w_vec / (k_rrf + Rank_vec(d))) + (w_bm25 / (k_rrf + Rank_bm25(d)))
```
* **Dynamic Noise Floor:** Samples pairwise similarity distributions. Cosine scores below the 50th-percentile noise floor are assigned zero evidential weight, filtering out random high-dimensional embedding collisions.

### 3.3 Stage 1: Rocchio Pseudo-Relevance Feedback (Vector Centroid Expansion)
To overcome vocabulary mismatch between conversational queries and stored episodic notes, the query vector is expanded toward the centroid of top first-pass candidates:
```
q_exp = alpha * q_0 + beta * (1 / |T|) * sum(d_i for d_i in T)
```
* **Default Hyperparameters:** `alpha = 0.7`, `beta = 0.3`, `|T| = 2`.
* The updated vector `q_exp` pulls the search into the exact semantic neighborhood of relevant historical passages.

### 3.4 Stage 2: Personalized PageRank (Bipartite Graph Activation)
Rather than naive 1-hop heuristics, Orma builds a bipartite Entity-State-Locus graph and runs power-iteration with random walks and restarts:
```
p^(t+1) = (1 - alpha_ppr) * seeds + alpha_ppr * (P_trans^T * p^(t) + dangling_mass)
```
* **Damping Factor:** `alpha_ppr = 0.85`, `iterations = 20`.
* **Teleportation Vector:** Seeded by entities and states identified in first-pass search.
* **Mass Conservation:** Dangling nodes (nodes with zero outgoing edges) distribute their mass uniformly back to the teleportation seeds.

### 3.5 Stage 3: FlashRank Neural Cross-Encoder Reranking
Top candidate passages retrieved by the expanded bi-encoder are scored using a CPU-optimized ONNX cross-encoder (`ms-marco-TinyBERT-L-2-v2`, ~3.2 MB):
```
Score_cross(q, d) = Softmax(W_cross * BERT(q [SEP] d))
Final_Score(d) = Score_cross(q, d) * Score_orig(d)
```
Full bidirectional cross-attention across all token pairs elevates needle-in-a-haystack evidence directly on CPU in ~3ms.

### 3.6 Stage 4: Maximal Marginal Relevance (Context Diversification)
To prevent duplicate conversational turns from monopolizing the token budget, candidate passages are selected greedily via MMR:
```
MMR = argmax_{d in C \ S} [ lambda * Rel(d) - (1 - lambda) * max_{s in S} CosSim(d, s) ]
```
* **Balance Parameter:** `lambda = 0.7`.
* High-relevance items with distinct phrasing are prioritized; near-duplicate turns are suppressed.

### 3.7 Cognitive Decay Dynamics
* **ACT-R Base-Level Decay:**
  ```
  Activation B_i = ln( sum_{k=1}^n (t - t_k)^(-d) )  [decay d = 0.5]
  ```
  Repeatedly accessed memories maintain high baseline strength; single-pass mentions fade gracefully.
* **Hawkes Conversational Self-Excitation:**
  ```
  lambda(t) = mu + sum_{t_i < t} alpha_h * exp(-beta_h * (t - t_i))
  ```
  Provides a 30-minute conversational half-life boost (`beta_h = ln(2) / 1800`) to items discussed within active sessions.
* **Ebbinghaus Spaced Stability Expansion:**
  ```
  Stability S_n = S_0 * (1 + gain * (n - 1))  [S_0 = 2.0 days, gain = 0.5]
  ```
  Each spaced retrieval expands memory half-life, ensuring established preferences resist decay for months.

---

## 4. Paninian Linguistics & Defeasible Epistemic Logic

Small 3B models fail at conversational memory because natural language is elliptical. Orma applies formal linguistic principles inspired by Panini's *Astadhyayi*:

### 4.1 Anuvṛtti (Pronoun Context Inheritance)
When a user says *"My sister Asha has a peanut allergy"* followed three turns later by *"She lives in Pune"*, naive models disconnect the pronoun or attach it to the user.
Orma's *anuvṛtti* resolution engine tracks active subject provenance:
* First-person pronouns (`I`, `me`, `my`) map strictly to the speaker.
* Third-person pronouns (`she`, `he`, `they`) carry forward the last explicitly bound named subject within the conversational episode window.
* Third-party attributes are isolated into separate entity records, preventing accidental attribution bleed.

### 4.2 Non-Monotonic Defeasible Logic
Real-world human preferences are non-monotonic:
* Baseline truth: *"I drink black coffee every morning at 7 AM."*
* Defeasible exception: *"I have a sore throat today, so I'm drinking ginger tea with honey."*

Naive memory systems detect a contradiction on `(user, morning_drink)` and overwrite the baseline habit.
Orma inspects the schema condition (`item.condition`):
* Categorical statements (`condition: ""`) trigger clean supersession (`active = 0, superseded_by_id = new_id`).
* Situational exceptions (`condition: "sore throat"`) are preserved alongside baseline habits as conditional overrides without closing the canonical state.

### 4.3 Inhibitory Control & Top-of-Prompt Hoisting
Autoregressive language models naturally activate on negative tokens (*"don't use semicolons"*).
* Orma identifies prohibited assertions (`prohibited: true` or `orma_forbid`).
* Prohibited rules are namespaced as `forbidden_<attribute>` and hoisted to the **very top** of the system prompt (`[FORBIDDEN: <rule>]`).
* Injected with dynamic negative framing semantics, yielding **100% compliance on small models without fine-tuning**.

---

## 5. Storage Schema & Data Durability

All memories are kept locally in **`~/.orma/memory.db`** with SQLite Write-Ahead Logging (WAL) enabled:

| Table | Role & Contents |
| :--- | :--- |
| `episodes` | Full, lossless append-only log of raw text, roles, speakers, sources, and timestamps. Never pruned. |
| `sentences` | Segmented sentences parsed from episodes, indexed for FTS5 and vector search. |
| `loci` | Spatial checkpoint rooms with embedded gist vectors (`vec BLOB`). |
| `entities` | Named subjects and objects (`user`, people, projects, places) with mention counters. |
| `entity_aliases` | Cross-referenced user nicknames and aliases mapped back to canonical entities. |
| `attributes` | Normalized attribute keys (`home_city`, `forbidden_semicolon`). |
| `states` | Single-valued assertions: subject, attribute, value, `valid_from`, `valid_to`, line citations. |
| `events` | Dated occurrences, participant IDs, and normalized occurrence timestamps. |
| `vec_*` | Virtual tables for `sqlite-vec` vector similarity search (384-dimensional float32). |
| `fts_*` | Virtual tables for `FTS5` BM25 full-text keyword indexing. |

---

## 6. Configuration & Ablation Switches

All parameters can be modified via `OrmaConfig` in Python or environment variables:

```python
from orma import Memory, OrmaConfig

config = OrmaConfig(
    enable_semantic=True,  # Background sleep consolidation
    enable_mmr=True,  # Maximal Marginal Relevance diversification
    mmr_lambda=0.7,  # 1.0 = pure relevance, 0.0 = pure diversity
    enable_rocchio=True,  # Rocchio PRF query expansion
    rocchio_alpha=0.7,  # Original query weight
    rocchio_beta=0.3,  # Feedback centroid weight
    rocchio_top_k=2,  # Passages used for feedback centroid
    enable_rerank=True,  # FlashRank ONNX neural cross-encoder
    rerank_model="ms-marco-TinyBERT-L-2-v2",  # Fast 3.2MB CPU ONNX model
    enable_spreading_activation=True,  # Personalized PageRank graph iteration
    enable_base_level=True,  # ACT-R power-law recency decay
    enable_hawkes=True,  # Hawkes conversational temporal burst
    enable_ebbinghaus=True,  # Ebbinghaus spaced stability scaling
    enable_defeasible_logic=True,  # Non-monotonic conditional exception logic
    enable_sutra_packing=False,  # Dense bracket encoding (-40% context tokens)
    budget_tokens=4096,  # Token ceiling for memory prompt blocks
)
mem = Memory(config=config)
```
