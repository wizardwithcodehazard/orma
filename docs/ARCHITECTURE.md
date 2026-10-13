# Architecture

An engineering overview of what is implemented. The design rationale is in [the spec](superpowers/specs/2026-10-12-local-neurosymbolic-memory-design.md), which is the source of truth for why things are built this way.

Principle: neural for perception, symbolic for memory. The model reads text into structure; code maintains the graph. No LLM call happens on the request path.

## Modules

**store** (`smruti/store/`). One SQLite file with typed tables plus sqlite-vec (vector) and FTS5 (lexical, porter tokenizer) indexes keyed by the same integer ids. `db.py` holds the schema, `records.py` the dataclasses (`Episode`, `Entity`, `Predicate`, `Fact`, `Event`, `Summary`, `Procedure`), `index.py` vector and text search, and `store.py` the `Store` class with all reads and writes. Every record carries a `space`.

**encode** (`smruti/encode.py`). Per-turn encoding, no LLM. Embeds the turn, measures novelty as 1 minus the highest similarity to existing episodes, facts and summaries, and stores it as an episode unless the adaptive gate says it is redundant. The gate drops a turn whose novelty is below the `novelty_percentile` of the space's own recent novelty scores (after `novelty_warmup` samples; window `novelty_window`). Dropped duplicates bump `seen_count` on the matching episode. Turns with a tool outcome are never dropped.

**recall** (`smruti/recall/`). Builds the prompt for a request with no LLM call and no writes. It fuses vector and lexical search over facts, events, procedures, summaries and pending episodes (reciprocal rank fusion, with hits weighted by calibrated similarity and query-term coverage), links query entities, spreads activation along current facts and to the events each entity took part in (`activation.py`, with the fan effect counting both), keeps the top `max_candidate_facts` facts and `max_candidate_events` events, scores relevance x recency x confidence (events: relevance x recency of when they happened), and packs into the token budget (`pack.py`). Memory is merged into the system prompt as a `## Memory` block with sections: known facts about the user (core), relevant facts, past events (`YYYY-MM-DD: gist`), lessons from past actions, background, recent notes (user and tool turns only, each cut to `recall_note_chars`), under a one-line framing that tells the model the memories are about the person it is talking to. The newest `min_recent_turns` turns are always kept; older turns fill leftover budget. Tokens are estimated as characters divided by `chars_per_token`.

**sleep** (`smruti/sleep/`). Background consolidation of pending episodes. `extract.py` is the neural half: the model returns entities, facts, events, procedures and summary updates as JSON in a fixed schema, with decoding constrained to `JSON_SCHEMA` when the runtime accepts `response_format` `json_schema` (else plain JSON mode). With `sleep_votes` >= 2 (default 1), `vote.py` marks the first-pass facts that a sampled replay of the same batch also produced (matched by embedding similarity of the rendered fact); every first-pass fact is still consolidated, and agreed ones count as one extra observation. `consolidate.py` is the symbolic half: it first drops facts, entities and user names whose terms (FTS tokenizer and porter stemmer) are not in the batch's log (the user's own names exempt), so facts copied from the prompt context cannot enter the graph; then it resolves names onto existing entities and predicates by embedding similarity, resolves the user's own names to `user` through aliases, closes a one-valued fact when a new value arrives (`valid_to` is set, the row is kept; a back-dated insert only when both times were stated, and a time on an episode's own UTC day does not count as stated), reinforces repeated facts, records procedures (only from batches that contain a tool turn) with success/failure evidence, records dated events (each needs at least one participant and every participant grounded; the date follows the same stated-time rule as facts; a retried batch adds no duplicate), and upserts summaries by topic. `Sleeper` batches episodes to fit `sleep_prompt_tokens`, prunes consolidated episodes older than `episode_retention_days` and stale low-confidence procedures.

**llm** (`smruti/llm.py`). A small client for any OpenAI-compatible `/chat/completions` endpoint. `chat_json` uses JSON mode at temperature 0 and retries once with a stricter reminder.

**memory** (`smruti/memory.py`). The SDK `Memory` class wiring store, encoder, recaller and sleeper together.

**serve** (`smruti/serve/`). `proxy.py` is a Starlette app. `POST /v1/chat/completions` runs recall, forwards upstream (streaming or not), relays the response, then records new turns in a background task. Every other `/v1/*` request is passed through unchanged. `capture.py` decides what to record: clients resend the whole conversation, so only the turns after the last assistant message are new (the user message and tool results, each tool result paired with the call that produced it). Replies are read from JSON bodies or accumulated from SSE streams. `scheduler.py` is a daemon thread that polls for due spaces and sleeps them.

**cli** (`smruti/cli.py`). Typer commands: `proxy`, `sleep`, `facts`, `recall`, `forget`, `stats`.

## Data model

| Table | Purpose |
| --- | --- |
| `episodes` | Raw turns: role, text, optional tool outcome, novelty, `seen_count`, `consolidated_at`, `sleep_attempts`, `sleep_error`. |
| `entities` | Named things (including the special `user`), with type and mention count. |
| `predicates` | Relation names with cardinality `one` or `many`. |
| `facts` | Subject, predicate, object entity or literal value, confidence, support, `valid_from`, `valid_from_stated`, `valid_to` (null while true), `last_used`, source episode ids. |
| `entity_aliases` | Lowercased alternative names per space and the entity they resolve to (the user's own names point at `user`). |
| `events` | Episodic memory: a gist of something that happened, `occurred_at`, `occurred_stated`, `created_at`, source episode ids. Never deleted. |
| `event_participants` | Which entities took part in each event. |
| `summaries` | One per (space, topic) with token count. |
| `procedures` | Trigger, action, error class, success and failure counts, source episode ids. |
| `sleep_leases` | One row per space: lease holder and expiry. |
| `novelty_stats` | Per-space window of recent novelty scores, used by the gate and to calibrate similarity. |
| `meta` | Schema metadata (embedding dimension). |

Alongside these are `vec_*` (sqlite-vec) and `fts_*` (FTS5) virtual tables for each searchable kind.

## Request path

1. The proxy picks the space (`X-Smruti-Space` header, else the body's `user`, else the default).
2. If sleep has no model yet, it adopts the request's `model`.
3. `Memory.build_context` runs recall and returns the messages to send.
4. The request goes upstream. The response is relayed as it arrives.
5. In a background task: for each new user or tool turn, `Memory.observe`, then the assistant reply. Only successful (status below 400) responses are recorded.

## Sleep path

A space is due when it has `sleep_after_episodes` pending episodes, or its newest pending episode is `sleep_idle_seconds` old. The scheduler polls every `--sleep-poll` seconds (default 30). Per space, `Sleeper.run` takes the lease, then loops: fetch up to `sleep_batch_episodes` pending episodes, trim to the prompt budget, build context (nearby entities, predicates and summaries so the model reuses names), extract `sleep_votes` times (temperature 0, then `sleep_vote_temperature`), vote, consolidate the first pass (grounded facts only, agreed ones reinforced once more), mark consolidated.

## Concurrency

One `Store` per `Memory`, one SQLite connection shared across threads, guarded by an `RLock`; write transactions are `BEGIN IMMEDIATE` and nest. The database is in WAL mode with a 5 s busy timeout. Sleep takes a per-space lease (`sleep_lease_seconds`, 600) so two sleepers never consolidate the same space; the lease is re-checked before each batch and released at the end.

## Failure handling

- The proxy fails open. If recall raises, the original request is forwarded unchanged. If recording raises, the error is logged and the response is unaffected.
- Upstream unreachable: the proxy returns 502 with an error body. Upstream error statuses are passed through and not recorded.
- A sleep batch that fails (model error, bad JSON after one retry, consolidation error) increments `sleep_attempts` on its episodes and stops that run; later cycles retry it. After `sleep_max_attempts` the episodes are retired (marked consolidated) so sleep cannot loop on a bad batch.
- Scheduler errors are logged and retried on the next poll.

## Ablation switches

`SmrutiConfig` flags, all default true, for measuring each component:

- `enable_novelty_gate`: off stores every turn.
- `enable_graph`: off skips entity linking and spreading activation.
- `enable_core_block`: off omits the "known facts about the user" block.
- `enable_similarity_weighting`: off weighs every similarity as 1 (plain RRF).
- `enable_episode_recall`: off omits unconsolidated episodes from recall.
- `enable_events`: off stores no events during sleep and recalls none.
