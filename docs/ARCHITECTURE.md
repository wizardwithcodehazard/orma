# Architecture

An engineering overview of Smruti v2. The design spec in `docs/superpowers/specs/` describes v1; where they differ, this file is current.

## Principle

Episodes are the memory; states are the index of what is true now.

Smruti borrows the shape of human memory and adapts it to a laptop:

| Brain | Laptop | Smruti |
|---|---|---|
| Episodes are stored fast; facts form slowly in sleep | Storage is cheap and lossless; the LLM is the scarce resource | Every record is kept in full, forever. Sleep spends one short LLM call per batch, and only on what needs structure |
| Semantic memory says what is true now | One-valued state is easy to maintain in code | Only **states** (one value at a time, superseded when it changes) and **dated events** are structured. Everything else is found as the sentences themselves |
| Recall is cue-driven and favours what is recent and often used | Search indexes are cheap | Hybrid vector and full-text search, with a bounded ACT-R boost for recency and repetition |
| Memories are reconsolidated | Data can be re-read by a better reader | `rebuild` re-reads every kept record, for example with a stronger model |

The language model is used in two places only: answering (the chat model) and reading batches during sleep. Identity, time, contradiction and ranking are code.

## A general memory layer

Any source enters the same way, like cognee's `add → cognify → search` or Mem0's `add / search`:

- `Memory.add(text, role=..., source=..., speaker=..., at=...)` and `Memory.add_file(path)`. Roles: `user`, `assistant`, `tool` (chat), `note` (the user's own words), `document`. Long text is kept as several records of about 2,000 characters, split between paragraphs.
- `Memory.observe(role, text)` is the chat adapter the proxy uses.
- `Memory.sleep()` consolidates; `Memory.rebuild()` forgets what sleep built and reads everything again.
- `Memory.search(query)` returns `{states, passages, events}`; `Memory.build_context(messages)` renders it into a chat prompt.
- CLI: `smruti add FILE|TEXT`, `smruti search QUERY`, `smruti sleep`, `smruti rebuild`, `smruti facts`, `smruti recall`, `smruti forget`, `smruti stats`, `smruti proxy`.

"I" means the user only in `user` and `note` records; a document or a tool is its own speaker.

## Modules

**text** (`smruti/text.py`). Structural helpers: sentence splitting at line breaks and sentence ends (not after initials or short abbreviations such as "Dr."), fenced code blocks reduced to one marker line, long sentences cut at words; whole-word matching; the words and word pairs a paraphrase is compared to.

**encode** (`smruti/encode.py`). Adding a record, no LLM. A record whose embedding is at least `novelty_duplicate_similarity` (0.97) similar to a kept record is counted (`seen_count`), not stored. The bar is high because embeddings barely see negation: bge-small scores "is allergic to" vs "is not allergic to" at 0.883. Everything else is kept and split into sentences, each embedded and indexed in vector and full-text search, so it can be found at once.

**sleep** (`smruti/sleep/`). The slow path, run when a space has `sleep_after_episodes` pending records or has been idle for `sleep_idle_seconds`.
- `__init__.py` batches pending records to fit `sleep_prompt_tokens` and shows their sentences as numbered lines, `[s42 2026-10-14 user] I moved to Bangalore last month.`; assistant sentences appear as `[context ...]` lines, cut to `context_chars`, never citable. A batch with nothing citable is marked read without a model call. Context: known names with aliases, attributes and current states near the batch.
- `extract.py` asks for two lists only, each item citing a line number: `states` (`about`, `attribute`, `value`, `ended`, `since`) and `events` (`what`, `when`, `who`), plus `user_name`. Citing by number keeps the answer short (the slow part on a laptop GPU) and provenance exact. Decoding is constrained to the JSON schema when the runtime supports it.
- `consolidate.py` checks and integrates:
  - **citation:** the line must be in the batch and written by a user, note, document or tool;
  - **who it is about** (the karta): "user", a claimed user name, or first-person words in the user's own lines mean the user; a third-person word ("she") carries over the last person named in the batch text before it (anuvrtti); any other name must occur in the batch or already be known. Pronoun lists are the config's `first_person_words` and `third_person_words`;
  - **grounding:** a state's value must occur in the cited line, or paraphrase a word or word pair of it (embedding similarity at least `entity_merge_threshold`);
  - **time:** one value per entity and attribute. A new value closes the current one (kept as history, `ended_by` the line that changed it); `ended` closes the matching value; a value with a stated date earlier than a current value's stated date is filed as history. Date words are resolved by `clock.py` against when the line was said, in the user's timezone; the day it was said is not a stated time;
  - attributes merge by name or by embedding (`attribute_merge_threshold`); entities by name, alias or embedding (`entity_merge_threshold`);
  - idempotent: a line already integrated adds nothing when a batch is retried.
- A failed batch is retried on later cycles and retired after `sleep_max_attempts`. Only one sleeper runs per space (a lease).

**recall** (`smruti/recall/`). The fast path: no LLM, no writes.
- Passages: hybrid search over sentences (vector and FTS5, fused by reciprocal rank, each similarity weighted against the space's own noise floor), score × a bounded boost for recency (`use_weight`, ACT-R base level). Only user, note, document and tool sentences; never those already in the conversation.
- States: the user's current states always; others' when the query shares words with them or names their holder.
- Events: the same hybrid search.
- A passage whose state has since changed is marked `[outdated: now ...]`.
- `pack.py` fills the token budget in priority order (states capped at `state_budget_share`, then passages, then events), keeps the newest `min_recent_turns` turns and fills leftover budget with older turns:

```
## Memory
What you remember about the person you are talking to ... Today is 2026-10-16.
Now true:
- user · home city: Bangalore (since 2026-09-16)
Things said before:
- 2026-10-15 user: I'm Sahil and I live in Pune. [outdated: now user · home city: Bangalore]
- 2026-10-16 user: Last Saturday I went trekking to Sinhagad with my sister Asha.
Events:
- 2026-10-10: user went trekking at Sinhagad with Asha
```

**clock** (`smruti/clock.py`). Calendar days in the user's timezone (`timezone`: empty for this machine's zone, `UTC`, or an IANA name), ISO parsing, and date words ("last Saturday", "3 days ago") resolved with dateparser against when they were said, preferring the past (`date_order` reads numeric dates such as 08/10/2026).

**serve** (`smruti/serve/`). `proxy.py`, an OpenAI-compatible Starlette proxy: recall into the system prompt, forward upstream (streaming or not), relay, then record the new turns through `observe`. `capture.py` decides which turns are new and masks secrets; `scheduler.py` runs sleep in the background.

**cli**, **memory**: the `smruti` command and the `Memory` SDK above.

## Data model

| Table | Purpose |
| --- | --- |
| `episodes` | Kept records: role, text, outcome, novelty, `seen_count`, `source`, `speaker`, `consolidated_at`, sleep attempts. Never deleted. |
| `sentences` | The sentences of each record, with role, speaker and time: what recall finds and sleep cites. |
| `entities`, `entity_aliases` | People and things named by states and events; the user's own names point at `user`. |
| `attributes` | Attribute names per space (`home_city`, `employer`, ...), merged by meaning. |
| `states` | Subject, attribute, value, `valid_from` (and whether it was stated), `valid_to` (null while true), `sentence_id` (the line that stated it), `ended_by`, `evidence` (every line that stated it) and their times. |
| `events`, `event_participants` | Dated happenings, the line that told them, and who took part. |
| `sleep_leases`, `novelty_stats`, `meta` | Sleep lease, the similarity distribution recall calibrates against, schema version and embedding dimension. |

Vector (`vec_*`, sqlite-vec) and text (`fts_*`, FTS5 with the Porter stemmer) indexes share ids with their tables. A version 1 database keeps its records; `smruti rebuild` splits them into sentences and consolidates them again.

## Concurrency and failure handling

One `Store` per `Memory`, one SQLite connection guarded by an `RLock`, WAL mode, `BEGIN IMMEDIATE` write transactions that nest. The proxy fails open: if recall raises, the request goes upstream unchanged; if recording raises, the reply is unaffected. Upstream errors are passed through and not recorded.

## Ablation switches

All default true:

- `enable_semantic`: off = no states or events; recall over kept sentences only. The baseline the rest must beat.
- `enable_novelty_gate`: off stores near-copies too.
- `enable_similarity_weighting`: off weighs every similarity as 1 (plain reciprocal rank fusion).
- `enable_base_level`: off ranks by match alone, without the recency boost.
- `enable_schema_decoding`: off uses plain JSON mode.
- `enable_grounding`: off accepts state values the cited line does not contain.
- `enable_date_phrases`: off accepts only ISO dates.
