# Orma (ഓർമ്മ)

> **Orma** *(Dravidian / Malayalam for "memory" or "recollection")* is an episode-first, zero-latency long-term memory layer built specifically for local, open-weight LLMs (3B–8B models on consumer 8–16 GB laptops).

[![PyPI Version](https://img.shields.io/pypi/v/orma-ai.svg)](https://pypi.org/project/orma-ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-197%20passed-brightgreen.svg)](tests/)

Orma can be used in two ways:
1. **As a native Python SDK (`from orma import Memory`)** for direct integration into your agents, chatbots, and apps (like Mem0 and Cognee).
2. **As an OpenAI-compatible proxy (`orma proxy` or `orma up`)** that transparently adds long-term memory to any LLM tool, client, or framework with zero code changes.

Everything runs locally in SQLite with `sqlite-vec` and fast on-device embeddings—**no external servers, no cloud API keys, and zero data loss.**

---

## Why Orma vs. Mem0 & Cognee?

Most agent memory frameworks (like Mem0 or Cognee) were designed for massive cloud models (GPT-4o, Claude 3.5 Sonnet) and enterprise graph databases:
1. **They freeze chat turns:** Calling an LLM synchronously on every `add()` causes 5–15 second freezes on local machines.
2. **They hallucinate graphs:** 3B models struggle with complex entity-relation extraction and emit malformed JSON or inverted relations.
3. **They permanently corrupt facts:** Prompting small models with destructive `DELETE / UPDATE` instructions leads to accidental data loss.

### The Orma Approach
* **Episodes are the memory:** Every conversation turn, document, or tool log is kept in full in an append-only log and indexed into sentences immediately.
* **Zero LLM calls on recall:** Recall runs in `< 25ms` using hybrid vector search (sqlite-vec) + BM25 full-text search (FTS5) with bounded recency boosting (ACT-R).
* **Sleep consolidates state asynchronously:** While the machine is idle, a background worker uses the local model to extract single-valued current states (`now true`) and dated events citing exact line numbers.
* **Deterministic code guards:** Time resolution, pronoun carry-over (*anuvṛtti*), and contradiction resolution are handled in verified Python code, not by trusting small model arithmetic.

| Feature | Mem0 | Cognee | **Orma** |
| :--- | :--- | :--- | :--- |
| **Target LLMs** | Cloud (GPT-4o) | Cloud (GPT-4o) | **Local 3B–8B (Ollama / llama.cpp)** |
| **Python SDK** | `Memory()` | `cognee.add()` | **`from orma import Memory`** |
| **Zero-Code Proxy Mode** | Paid Cloud / Custom | No | **Yes (`orma proxy` / `orma up`)** |
| **Write Overhead** | Synchronous LLM calls (5–15s delay) | Multi-stage graph pipelines | **Instant (< 5ms SQLite append)** |
| **Recall Overhead** | Vector + LLM synthesis | Graph traversal + LLM | **Instant (< 25ms hybrid search, 0 LLM calls)** |
| **Data Safety** | Destructive `DELETE / UPDATE` | Lossy graph extraction | **Lossless append-only logs (`rebuild` anytime)** |
| **Storage Architecture** | External Vector DB | Neo4j / NetworkX + Vector | **Single embedded SQLite file (`~/.orma/memory.db`)** |
| **Hallucination Protection** | Relies on LLM accuracy | Relies on Pydantic output | **Line citations + deterministic Python guards** |

---

## Architecture & Data Flow

### 1. Online Recall & Cognitive Retrieval Pipeline (< 25ms, Zero LLM Calls)

```mermaid
flowchart TD
    User["User / Agent / Client"] -->|"1. Chat Request"| Proxy["Orma Proxy / SDK"]
    
    subgraph MultiSearch ["Multi-Channel Hybrid Search (SQLite)"]
        Proxy -->|"2a. BM25 Query"| FTS["FTS5 BM25 Exact Text Index"]
        Proxy -->|"2b. Vector Query"| Vec["sqlite-vec (bge-small-en-v1.5)"]
        FTS --> RRF["Reciprocal Rank Fusion (RRF)"]
        Vec --> RRF
        RRF --> NoiseFloor["Adaptive Space Noise Floor Calibration"]
    end

    subgraph SpreadingGraph ["ACT-R Spreading Activation (SQLite Entity Graph)"]
        NoiseFloor --> ExtractEntities["Extract Seed Entities from Query & States"]
        ExtractEntities --> Adjacency["Traverse Bi-directional Entity Links"]
        Adjacency --> FanEffect["Anderson's Fan Effect: Weight / Fan(Entity)"]
        FanEffect --> AssociateStates["Boost Associated Context (Dr. Rao -> Clinic -> Toothache)"]
    end

    subgraph CognitiveRanker ["Stochastic & Cognitive Scoring Suite"]
        AssociateStates --> Hawkes["Hawkes Self-Exciting Temporal Burst: α · exp(-β · Δt)"]
        Hawkes --> Ebbinghaus["Ebbinghaus Spaced Stability: S = S0 · (1 + gain · (n - 1))"]
        Ebbinghaus --> BaseLevel["ACT-R Base-Level Decay: Σ (age_days + 1)^(-d)"]
    end

    subgraph BudgetPacker ["Token-Budget Packing & Guardrails"]
        BaseLevel --> Novelty["Novelty Gate: Drop duplicates (cosine >= 0.97)"]
        Novelty --> Partition["Partition Budget: Now True | Outdated History | Events | Sentences"]
        Partition --> MemoryBlock["Construct Formatted System Memory Prompt Block"]
    end

    MemoryBlock -->|"3. Injected Prompt"| Upstream["Local LLM (Ollama / llama.cpp / vLLM)"]
    Upstream -->|"4. Streaming Response"| Proxy
    Proxy -->|"5. Return to User"| User
    Proxy -.->|"6. Async Non-Blocking Append"| EpisodicLog[("SQLite Append-Only Episodes")]
```

### 2. Background Sleep Consolidation & Defeasible Epistemic Logic

```mermaid
flowchart TD
    EpisodicLog[("Pending SQLite Episodes")] -->|"Idle (120s) or Batch Threshold"| Batcher["Episode Batcher (Line IDs [s1, s2, ...])"]
    Batcher --> SleepLLM["Local Sleep Model (Qwen2.5-3B / Llama-3.2-3B)"]
    SleepLLM -->|"Constrained JSON Schema Output"| RawExtraction["Raw States & Events with Line Citations"]

    subgraph DeterministicVerification ["Deterministic Neuro-Symbolic Verification"]
        RawExtraction --> V1["Line Citation Check (Cited sentence exists?)"]
        V1 --> V2["Grounding Engine (Value matches / paraphrases line?)"]
        V2 --> V3["Pronoun & Anaphora Resolution (Anuvṛtti tracking)"]
        V3 --> V4["Calendar & Timezone Anchoring (clock.py)"]
    end

    subgraph DefeasibleLogic ["Defeasible Epistemic Logic Gate"]
        V4 --> FrameCheck{"Has Conditional Framing? (when, if, unless, during)"}
        FrameCheck -->|"No (Categorical Fact)"| Categorical["Supersedes Contradictory Prior State"]
        Categorical --> CloseOld["Mark Old State: active = 0, superseded_by_id = new_id"]
        FrameCheck -->|"Yes (Defeasible Exception)"| Defeasible["Preserve Baseline Truth + Record Conditional Override"]
    end

    CloseOld --> Store[("SQLite State Store & Entity Adjacency Graph")]
    Defeasible --> Store
```

---

## Installation

```bash
pip install orma-ai
```

Or install from source for development:
```bash
git clone https://github.com/wizardwithcodehazard/orma.git
cd orma
pip install -e ".[dev]"
python -m pytest -q  # 197 tests pass locally
```

---

## Usage Option 1: Python SDK (Direct In-Code Integration)

Use the Python SDK to embed Orma directly into your agents, pipelines, or web applications:

```python
from orma import Memory

# 1. Initialize memory (defaults to ~/.orma/memory.db)
mem = Memory(
    upstream="http://localhost:11434/v1",  # Ollama or llama.cpp endpoint
    model="llama3.2:3b"                     # Model for background sleep consolidation
)

# 2. Add memories, documents, or ongoing chat turns
mem.add("I work as a backend engineer and I live in Pune.")
mem.add_file("project_notes.md")

# Record chat interactions
mem.observe(role="user", text="I moved from Pune to Bangalore last week.")
mem.observe(role="assistant", text="Congratulations on the move to Bangalore!")

# 3. Retrieve formatted context for prompt injection
messages = [{"role": "user", "content": "Where do I currently live?"}]
prompt_messages = mem.build_context(messages)
# prompt_messages contains a system prompt with current facts, outdated facts marked, and relevant quotes!

# 4. Search memory directly
results = mem.search("Where is my home?")
for state in results.states:
    print(state.text)  # "user · home city: Bangalore"

# 5. Background consolidation (extracts facts and closes outdated ones)
report = mem.sleep()
print(report)

mem.close()
```

---

## Usage Option 2: Zero-Code Proxy (OpenAI SDK, LangChain, Cursor, Open WebUI)

Start Orma as a local proxy:
```bash
# With Ollama
orma proxy --model llama3.2:3b

# Or with llama.cpp
orma proxy --upstream http://127.0.0.1:9931 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
```

Then point any client to `http://localhost:8080/v1`:

### With the Official OpenAI Python Client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "What is my job and where do I live?"}]
)
print(response.choices[0].message.content)
```

### With LangChain:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",
    model="llama3.2:3b"
)
```

### With Open WebUI, AnythingLLM, or Cursor:
* Set OpenAI Base URL to: `http://localhost:8080/v1`
* Set API Key to: `none`

---

## Usage Option 3: One-Command Web UI (`orma up`)

If you are using llama.cpp or Ollama with a browser web UI, run:

```bash
orma up --upstream http://127.0.0.1:9931 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
```

This immediately:
1. Starts the Orma memory proxy on port 8080.
2. Automatically opens your default web browser to `http://127.0.0.1:8080/#/`.
3. Passes through the web UI frontend while transparently routing all chat completions through Orma's memory layer.

---

## Multi-User & Multi-Tenant Isolation

Orma supports isolated memory spaces out of the box so multiple users or agent sessions don't contaminate each other's memories:

* **In Python SDK:**
  ```python
  mem.add("Alice likes green tea.", space="user_alice")
  context = mem.build_context(messages, space="user_alice")
  ```
* **In HTTP Requests / Proxies:**
  Pass the `x-orma-space` header or `"user"` field:
  ```bash
  curl http://localhost:8080/v1/chat/completions \
    -H "x-orma-space: user_123" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Who am I?"}]}'
  ```

---

## CLI Reference

Orma provides a command-line tool for managing your memory:

```bash
# Add a note or document
orma add "My dentist is Dr. Rao in Kothrud."
orma add docs/handbook.md

# Search memory
orma search "dentist"

# Inspect active facts and outdated history
orma facts --history

# Preview the exact prompt block that would be injected
orma recall "Who is my dentist?"

# Manually trigger sleep consolidation
orma sleep --model llama3.2:3b

# Forget a specific fact
orma forget 12

# Database statistics
orma stats
```

---

## Memory Prompt Output Format

When Orma recalls memories, it packs them into a token-budgeted block injected into the system prompt:

```markdown
## Memory
What you remember about the person you are talking to. Today is 2026-10-24.

Now true:
- user · home city: Bangalore (since 2026-10-17)

Things said before:
- 2026-10-10 user: I'm Sahil and I live in Pune. [outdated: now user · home city: Bangalore]
- 2026-10-18 user: Last Saturday I went trekking to Sinhagad with my sister Asha.

Events:
- 2026-10-12: user went trekking at Sinhagad with Asha
```

---

## Cognitive & Mathematical Architecture

Orma integrates established principles from cognitive science, stochastic point processes, and non-monotonic logic directly into its zero-latency retrieval engine:

```
+---------------------------------------------------------------------------------------------------+
|                                 ORMA COGNITIVE RETRIEVAL PIPELINE                                 |
+------------------------------------+--------------------------------------------------------------+
| Hybrid Search (BM25 + sqlite-vec)  | Multi-channel fusion calibrated against dynamic noise floor  |
| Spreading Activation (ACT-R)       | Associative entity graph traversal with Fan Effect damping   |
| Hawkes Temporal Burst Process      | Self-exciting exponential kernel for conversational momentum |
| Ebbinghaus Retention & Stability   | Dynamic stability scaling per spaced repetition & rehearsal  |
| Defeasible Epistemic Logic         | Conditional exception handling preventing false overwrites   |
+------------------------------------+--------------------------------------------------------------+
```

---

### 1. Hawkes Self-Exciting Temporal Process (`enable_hawkes`)

**Cognitive Rationale:** Human dialogues exhibit natural clustering and temporal burstiness (Hawkes 1971, Kleinberg 2002). When a user brings up a topic (e.g., an ongoing project or medical symptom), subsequent exchanges revolve around that cluster. Standard recency decay either forgets past topics too fast or keeps them active too long.

**Mathematical Formulation:**
```
Intensity(t) = alpha * exp(-beta * dt)
beta = ln(2) / half_life_seconds
```

* **Burst Amplitude (`alpha = 0.4`)**: Boosts items accessed in active conversational bursts.
* **Decay Constant (`beta = ln(2) / 1800`)**: A 30-minute half-life ensures topics discussed earlier in a session maintain strong conversational momentum without lingering indefinitely into future days.
* **Bounded Ranking**: Implemented in [orma/recall/rank.py](file:///c:/Users/Sahil/Desktop/memory_layer/smruti/orma/recall/rank.py) as a bounded additive boost so temporal bursts never override exact factual matches.

---

### 2. Ebbinghaus Spaced Repetition Stability (`enable_ebbinghaus`)

**Cognitive Rationale:** The human forgetting curve (Ebbinghaus 1885, Cepeda et al. 2008) demonstrates that memory retention decays exponentially, but each spaced retrieval increases the memory's underlying **stability** (half-life). A fact stated once in passing should gracefully fade, whereas a fact confirmed across 4 different days should remain permanently accessible.

**Mathematical Formulation:**
```
Retention R(dt) = exp(-dt / S)
Stability S(n)  = S0 * (1 + gain * (n - 1))
```

* **Baseline Stability (`S0 = 2.0` days)**: Initial stability for a single mention.
* **Rehearsal Gain (`gain = 0.5`)**: For every subsequent citation or spaced rehearsal `n`, memory stability expands linearly, lengthening the half-life before decay.
* **Dual Protection**: Single-pass noise drops out of recall candidate pools, while repeated preferences resist decay for months.

---

### 3. ACT-R Spreading Activation & Anderson's Fan Effect (`enable_spreading_activation`)

**Cognitive Rationale:** Human memory recalls information associatively (Anderson 1983). When asked *"Where should I go for my toothache?"*, the query may not contain the doctor's name, but the concept of *toothache* is associatively linked to *dentist* and *Dr. Rao*. Traditional graph databases (e.g. Neo4j) add 50–150ms of network overhead. Orma executes associative graph traversal directly within SQLite using indexed adjacency tables.

**Mathematical Formulation:**
```
Activation A_i = B_i + sum_j (W_j * S_ji)
Association S_ji = Base_Weight / Fan(Entity_j)
```

* **Source Seed Extraction**: Seed entities are identified from hybrid text matches and active states.
* **The Fan Effect**: John R. Anderson's Fan Effect proves that associative activation spreading from a concept is inversely proportional to the number of facts associated with it:
  * A specific entity (e.g., `"Dr. Rao"` with `Fan = 2`) channels 50% of its associative energy directly to the target clinic state.
  * A general hub (e.g., `"Pune"` with `Fan = 60`) dissipates its energy across dozens of states, preventing generic concepts from hijacking the recall budget.

---

### 4. Defeasible Epistemic Logic (`enable_defeasible_logic`)

**Cognitive Rationale:** Real-world beliefs are non-monotonic. When a user states:
> *"I drink black coffee every morning."*
> followed later by:
> *"When I have a sore throat, I drink ginger tea with honey."*

Naive memory systems (like Mem0 or basic graph extraction) detect a contradiction on the entity-attribute `(user, drink)` and destructively overwrite the baseline habit.

**Epistemic Rule Engine:**
* Orma inspects assertions using semantic and syntactic conditional markers (`when`, `if`, `unless`, `during`, `in case of`).
* **Categorical Assertions** (`"I moved to Bangalore"`): Trigger state supersession (`active = 0, superseded_by_id = new_state_id`).
* **Defeasible Overrides** (`"When sick, drink tea"`): Registered as conditional exceptions without closing or corrupting the baseline categorical truth. Both facts are preserved in context.

---

### 5. Adaptive Noise Floor Calibration & Reciprocal Rank Fusion

**Algorithmic Rationale:** Small embedding models (e.g., 384-dimensional `bge-small-en-v1.5`) suffer from geometric concentration on high-dimensional unit hyperspheres—unrelated sentences frequently score 0.45–0.55 cosine similarity by pure random chance.

* **Noise Floor Estimation**: Orma dynamically samples pairwise distances to compute the 50th-percentile noise floor of the embedding space.
* **Similarity Floor Truncation**: Scores below the space noise floor carry zero evidential weight.
* **Reciprocal Rank Fusion (RRF)**:
  ```
  RRF Score = (1 / (60 + Rank_BM25)) + (1 / (60 + Rank_Vector))
  ```
  Fuses exact keyword recall (FTS5 BM25) with semantic vector search (sqlite-vec) with zero hyperparameter tuning.

---

## Storage & Data Durability (100% Local & Persistent)

* **Single File SQLite Storage**: All memories reside in `~/.orma/memory.db` with Write-Ahead Logging (WAL) enabled.
* **Survives Restarts & Power Cycles**: You can shut down `orma up`, restart your machine, or close your terminal at any time; your complete memory state, episodic history, and consolidated facts remain permanently stored.
* **Lossless Append-Only**: Episodes are never destructively overwritten or pruned by hallucinating models. You can inspect or re-consolidate your history anytime with `orma facts --history`.

---

## Configuration & Ablation Switches

Every mechanism in Orma can be independently configured or ablated in [orma/config.py](file:///c:/Users/Sahil/Desktop/memory_layer/smruti/orma/config.py) or passed via `OrmaConfig`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enable_semantic` | `True` | Background consolidation into states & events (off = pure episodic sentence search baseline) |
| `enable_novelty_gate` | `True` | Drops duplicate sentences (cosine similarity >= 0.97) |
| `enable_similarity_weighting` | `True` | Calibrates vector similarity against adaptive space noise floor |
| `enable_base_level` | `True` | ACT-R recency and frequency power-law decay |
| `enable_hawkes` | `True` | Hawkes self-exciting temporal process for conversational bursts |
| `enable_ebbinghaus` | `True` | Ebbinghaus spaced repetition stability scaling |
| `enable_spreading_activation` | `True` | Associative spreading activation with Fan Effect |
| `enable_defeasible_logic` | `True` | Non-monotonic conditional exception protection |
| `budget_tokens` | `4096` | Token budget ceiling for injected memory blocks |
| `search_k` | `30` | Top-k candidates pulled per hybrid search channel |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant for combining BM25 and vector scores |

---

## License

Distributed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for more information.
