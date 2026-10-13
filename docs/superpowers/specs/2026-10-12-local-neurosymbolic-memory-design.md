# Smruti v1 — Local Neuro-Symbolic Memory Layer: Design

Date: 2026-10-12 · Status: approved in brainstorming, pending spec review

## 1. Goal

Give small local LLMs (reference: 3B model, 4k-token budget, 8–16 GB laptop) long-term memory so they behave as if they had a much larger context window. Smruti sits between any chat client and any OpenAI-compatible runtime (Ollama, llama.cpp server, LM Studio, vLLM), and remembers across conversations with no code changes for the user.

It is also a research artifact. The research question:

> Does outcome-grounded, consolidated neuro-symbolic memory let a 4k-context 3B model match or beat larger contexts and retrieval-only memory on long-term memory benchmarks, at lower per-turn cost?

### Success criteria
1. `smruti proxy` works in front of Ollama with Open WebUI or a plain OpenAI client, streaming included.
2. Per-request overhead is under 60 ms on the reference laptop, with zero LLM calls on the request path.
3. A LongMemEval harness reports accuracy and tokens per turn for smruti against three baselines (§8).
4. Every tunable is in one config object and every component can be disabled, so ablations are possible.

### Non-goals (v1)
Multi-user servers and auth, Postgres, team sync, cloud LLM providers as the sleep model, image or audio memory.

## 2. Principle: neural for perception, symbolic for memory and reasoning

- **Neural**: embeddings (novelty, similarity, entity linking) and the local LLM during sleep (reading text into a fixed JSON schema).
- **Symbolic**: the typed schema, the entity–fact graph, temporal validity, contradiction resolution, evidence-based confidence, decay, and budgeted packing.
- **No hand-written text heuristics**: no regexes over natural language, no keyword or verb lists, no alias tables, no fixed magic thresholds where a data-adaptive one is possible. Remaining constants live in `SmrutiConfig`.

## 3. Architecture

| Module | Responsibility | Neural / symbolic |
|---|---|---|
| `smruti/store/` | SQLite schema and migrations, FTS5 and sqlite-vec indexes, typed repositories | Symbolic |
| `smruti/encode.py` | Per turn: embed, adaptive novelty gate, store or count | Neural + statistics |
| `smruti/recall/` | Per request: entity linking, graph expansion, hybrid ranking, budgeted packing | Neural linking → symbolic lookup and packing |
| `smruti/sleep/` | Idle-time: LLM extraction into a schema, then symbolic consolidation | Neural extraction → symbolic consolidation |
| `smruti/llm.py` | Minimal OpenAI-compatible client (chat, JSON mode, streaming) | — |
| `smruti/serve/` | `proxy` (OpenAI-compatible server), Python SDK, CLI, MCP, agent hooks | Interface |
| `smruti/eval/` | LongMemEval harness and baselines | — |

Package layout is a clean rewrite. The previous engine (`engine/`, `framework.py`, `storage/backends.py`, `storage/sync.py`) is removed; git history keeps it.

### Flow
- **Each request:** recall builds the prompt → upstream LLM → reply streamed back to the client → encode stores the user turn and the reply.
- **Idle:** sleep runs when a space has ≥ `sleep_after_episodes` (default 20) unconsolidated episodes, or has been idle for `sleep_idle_seconds` (default 120). It runs in a background thread in the proxy, or on demand with `smruti sleep`.

## 4. Storage (single SQLite file per data directory)

Every row carries a `space` column (default `"default"`). A space is a memory namespace, selected per request.

| Table | Columns (key ones) | Notes |
|---|---|---|
| `episodes` | id, space, role (`user`/`assistant`/`tool`), text, outcome (`null`/`success`/`failure`), embedding, novelty, seen_count, created_at, consolidated_at | Raw turns. Deleted `episode_retention_days` (7) after consolidation. |
| `entities` | id, space, name, type, embedding, mention_count, last_seen | Graph nodes. The `user` entity (the person chatting) always exists. |
| `predicates` | id, space, name, cardinality (`one`/`many`), embedding | Normalized relation vocabulary, learned during sleep. |
| `facts` | id, space, subject_id, predicate_id, object_id (nullable), object_value (nullable), confidence, support, valid_from, valid_to, last_used, source_episode_ids | Graph edges. Current facts have `valid_to IS NULL`. Never deleted; closed. |
| `summaries` | id, space, topic_entity_id (nullable), text, embedding, period_start, period_end, tokens | One rolling summary per topic, rewritten each sleep. |
| `procedures` | id, space, trigger, action, error_class, successes, failures, created_at, last_used | "When X → do / avoid Y". Polarity is the majority outcome (works if successes ≥ failures). Covers coding-agent failure memory. |
| `novelty_stats` | space, recent_scores (JSON ring buffer of last 500) | For the adaptive novelty gate. |

Indexes: FTS5 over `episodes.text`, `facts` (rendered text), `summaries.text`, `procedures.trigger`. sqlite-vec tables for episodes, entities, predicates, summaries, procedures (384-dim, `BAAI/bge-small-en-v1.5` via fastembed).

**Graph**: entities are nodes and facts are edges. Traversal uses recursive CTEs over current facts (`valid_to IS NULL`) up to `max_hops` (default 2).

## 5. Encode (per turn, no LLM, target < 20 ms)

1. Embed the text.
2. novelty = 1 − max cosine similarity to the space's 50 nearest episodes, facts and summaries.
3. Adaptive gate: during warm-up (< 50 scores in the space) always store. Afterwards, if novelty < the `novelty_percentile` (10th) of the space's recent scores, do not store; increment `seen_count` of the nearest episode instead. Always append the score to the ring buffer.
4. Store the episode. Tool episodes carry `outcome`.

## 6. Recall (per request, no LLM, target < 40 ms)

Input: space, incoming messages, token budget (default `budget_tokens` 4096 minus the client's system prompt and current user message).

0. **Similarity calibration** (once per request): raw cosine similarity is not evidence by itself (small embedders give unrelated texts ~0.5). The space's novelty buffer holds each incoming turn's nearest-neighbour novelty, so `1 - novelty` is the empirical similarity distribution of real traffic; the floor is its `similarity_floor_percentile` (default 50: the median, the closeness an arbitrary turn's nearest memory reaches by chance). Calibrated weight `w(sim) = clip((sim - floor) / (1 - floor), 0, 1)`. During warm-up (fewer than `novelty_warmup` scores) or with `enable_similarity_weighting` off, `w = 1`.
1. **Entity linking**: embed the current user message; KNN over `entities` (top `entity_link_k` 8, cue weight = `w(sim)`, linked only if `w > 0`); plus a lexical mention: an FTS5 entity hit links with weight 1.0 only if *all* of the entity name's terms occur in the message (symbolic containment over the FTS tokenizer's own porter-stemmed terms, so "any new ideas?" does not mention "New York"). `user` is not linked unconditionally: it is a hub and would otherwise dominate.
2. **Spreading activation with the fan effect (ACT-R)**: cues are the linked entities plus the subject/object entities of the search-hit facts (weight = that fact's fused score, max per entity). For `max_hops` steps, each frontier entity e with activation a gives every current fact incident to e `a × hop_decay^step / fan(e)`, where fan(e) is e's number of current facts; the fact's other endpoint entity carries that amount into the next frontier (each entity visited once). A hub's activation is thus split thinly across its many facts, while a specific entity lends its few facts a lot.
3. **Hybrid search**: FTS5 + vector KNN over facts, summaries and procedures, with weighted reciprocal-rank fusion: an item at rank r contributes `weight / (rrf_k + r + 1)` (`rrf_k` 60), normalized by the best possible score `len(rankings) / (rrf_k + 1)`; items with total 0 are dropped. Vector weight = `w(sim)`; FTS weight = the fraction of the query's distinct (stemmed) terms the row contains, so a hit matching one word of a long query counts little.
4. **Score** = relevance × recency (exp decay on `last_used`, half-life `recency_half_life_days` 14) × confidence. For facts, relevance = fused search score + spread activation (`hop_decay` default 0.6); with `enable_graph` off it is the search score alone. Only the `max_candidate_facts` (64) most relevant facts are scored, so a long tail of faintly activated facts cannot starve procedures and summaries.
5. **Pack**, in this fixed order, until the budget is used:
   1. *Core block*: current facts about `user` with confidence ≥ `core_min_confidence` (0.7), sorted by (predicate, id) for a stable, cache-friendly prefix.
   2. Relevant facts, then procedures, then summaries, by score.
   3. Recent raw turns from the client's messages, newest first; the last 2 are always kept.
6. Rendered as a `## Memory` section appended to the client's system prompt (or sent as the system message if there is none). Recall is read-only: `last_used` changes only when new evidence arrives, so retrieval never reinforces memory and the request path does no writes.

Token counting: the upstream `/tokenize` endpoint when available (llama.cpp), else `len(text) / chars_per_token` (default 3.5, configurable).

## 7. Sleep (idle, uses the same local LLM)

Per space, over unconsolidated episodes in batches of `sleep_batch_episodes` (12):

1. **Extract (neural)**: one JSON-mode call per batch. The prompt provides the batch text, the names of currently linked entities, the existing predicate vocabulary, and the output schema:
   ```json
   {"entities": [{"name": "", "type": ""}],
    "facts": [{"subject": "", "predicate": "", "object": "", "cardinality": "one|many", "time": "optional ISO date"}],
    "procedures": [{"trigger": "", "action": "", "polarity": "works|fails", "error_class": ""}],
    "summary_updates": [{"topic": "", "text": ""}]}
   ```
   Invalid JSON is retried once at temperature 0, then the batch is skipped and left unconsolidated.
2. **Consolidate (symbolic)**:
   - **Entities**: merge into an existing entity when embedding similarity ≥ `entity_merge_threshold` (0.85) and the type matches; else create.
   - **Predicates**: normalize to an existing predicate when similarity ≥ `predicate_merge_threshold` (0.8); else create with the extracted cardinality.
   - **Facts**: the same (subject, predicate, object) already current → `support += 1`, confidence recomputed. On a `one`-cardinality predicate, a different object closes the previous current fact (`valid_to = time or now`) and inserts the new one. `many` → insert.
   - **Confidence** = support / (support + 1) for facts; Beta mean (successes + 1) / (successes + failures + 2) for procedures.
   - **Procedures**: merge by trigger and action similarity ≥ 0.85, adding to successes or failures by polarity.
   - **Summaries**: replace the topic's summary with the extracted text; cap at `summary_max_tokens` (200).
3. Mark the batch consolidated; delete episodes older than the retention period; decay procedures (not facts) unused for `procedure_ttl_days` (30) and below confidence 0.5.

## 8. Evaluation (`smruti/eval/`)

- **Dataset**: LongMemEval-S (downloaded from Hugging Face on first run, cached). `--limit N` for subsets; the default dev run uses a stratified 50-question subset across the question types.
- **Systems**: `smruti` (full); `last-k` (most recent turns that fit 4k); `full-context` (truncated to the model's max context); `naive-rag` (vector top-k raw turns into 4k).
- **Ablations**: no graph expansion, no novelty gate, no sleep (episodes only), no core block.
- **Metrics**: accuracy per question type, judged by a configurable OpenAI-compatible judge endpoint (default: same local server, larger model if available); prompt tokens per answer; recall latency; sleep LLM calls per 100 turns.
- **Output**: JSON results plus a Markdown table in `eval_results/`.

## 9. Interfaces

- **Proxy**: `smruti proxy --upstream http://localhost:11434/v1 --port 8080`. Implements `POST /v1/chat/completions` (stream and non-stream) and passes `GET /v1/models` through. Space comes from the `X-Smruti-Space` header, else the request `user` field, else `default`.
- **SDK**: `Memory(path, upstream)` with `.chat(messages)`, `.build_context(messages, budget)`, `.observe(role, text, outcome=None)`, `.sleep()`, `.facts(entity)`, `.forget(fact_id)`.
- **CLI**: `init`, `proxy`, `sleep`, `facts`, `recall`, `forget`, `stats`, `eval`, `hook` (agent tool episodes with outcome), `run`.
- **MCP**: `recall`, `observe`, `facts` tools for agent clients.

## 10. Error handling
- The proxy fails open: any memory error → log it and forward the original request untouched.
- The upstream is unreachable → return the upstream error to the client unchanged.
- A sleep failure on a batch leaves it unconsolidated; it is retried next cycle; after 3 failures the batch is marked `consolidated` with an error note to avoid loops.
- SQLite runs in WAL mode with one writer thread in the proxy; readers use separate connections.

## 11. Testing
- Unit tests per module with a fake LLM (deterministic JSON fixtures) and a real SQLite file in a temporary directory.
- Property tests for consolidation: a contradiction on a `one` predicate always leaves exactly one current fact; support never decreases.
- Proxy integration test against a fake upstream (non-streaming and streaming).
- Eval smoke test on 3 bundled synthetic questions.
- Latency test asserting recall < 60 ms on 10k facts (marked slow).
