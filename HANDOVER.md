# Smruti handover

Everything you need to take over Smruti: what it is, how to run it, how it is built and why, what has been verified, what is broken, and what to do next.

## 1. What it is, in one paragraph

Smruti is a general long-term memory layer for small local LLMs (reference: Qwen2.5-3B on llama.cpp or `llama3.2:3b` on Ollama, 4k-token budget, an 8 GB laptop). Anything can be added: chat turns, notes, documents, tool output. Every record is kept in full and is searchable at once. In the background, while the machine is idle, the same local model reads new records and builds two kinds of structure that plain search cannot give: **states** (what is true now, one value at a time, superseded when it changes, with history) and **dated events**. Recall needs no LLM: hybrid search over the kept sentences plus the current states, rendered into the prompt with outdated lines marked. It ships as an SDK (`Memory.add / search / sleep / build_context`), a CLI, and an OpenAI-compatible proxy that adds memory to any chat client.

## 2. Run it (about 5 minutes)

Python 3.10 or newer.

```bash
git clone https://github.com/wizardwithcodehazard/smruti.git && cd smruti
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                                     # no downloads needed
```

```bash
python -m smruti add "My dentist is Dr. Rao in Kothrud."     # a note
python -m smruti add notes/handbook.md                       # a document
python -m smruti search "who is my dentist?"
python -m smruti sleep --model MODEL --upstream http://127.0.0.1:PORT/v1
python -m smruti facts --history

# the chat proxy (Ollama default; llama.cpp: --upstream http://127.0.0.1:PORT/v1)
python -m smruti proxy --model llama3.2:3b
```

`python -m smruti` always runs this checkout, even if another `smruti` command is first on PATH. The first run downloads the bge-small embedding model (about 130 MB). Memory lives in `~/.smruti/memory.db` unless `SMRUTI_DIR` is set. A database from version 1 keeps its records; run `smruti rebuild` once.

## 3. Repo map

| Path | What it is |
|---|---|
| `smruti/config.py` | `SmrutiConfig`: every tunable and every ablation switch (`enable_*`) |
| `smruti/text.py` | Sentence splitting, code-block markers, whole-word matching |
| `smruti/clock.py` | Timezone-aware days; date words resolved against when they were said |
| `smruti/store/` | SQLite schema (`db.py`), vector and FTS5 helpers (`index.py`), records, `Store` |
| `smruti/encode.py` | Adding a record: near-copy gate, sentences, indexing |
| `smruti/sleep/` | Batching and the lease (`__init__.py`), the two-list extraction (`extract.py`), checks and time maintenance (`consolidate.py`) |
| `smruti/recall/` | Hybrid search, ranking (`rank.py`), budgeted packing (`pack.py`) |
| `smruti/serve/` | The proxy, turn capture, the background sleep scheduler |
| `smruti/memory.py`, `smruti/cli.py` | The SDK and the `smruti` command |
| `eval/` | Dev cases (`cases.json`), held-out test cases (`test_cases.json`), the harness (`run.py`, `score.py`) |
| `tests/` | Unit and integration tests with `FakeEmbedder`, `NoisyEmbedder`, `FakeLLM` and `seed_state` |
| `scripts/` | Real-model chat demo, latency and needle-in-a-haystack smoke test |
| `docs/ARCHITECTURE.md` | Engineering overview of v2 (current) |
| `docs/superpowers/` | The v1 design spec and plans (historical) |

## 4. How it works

Read `docs/ARCHITECTURE.md`. In short:

- **add** (no LLM): keep the record, split into sentences, embed and index them. A near-copy (≥ 0.97) is counted, not stored.
- **sleep** (one LLM call per batch): numbered lines in, `{user_name, states, events}` out, each citing a line number. Code checks the citation, who it is about (pronouns carry over from the last name before them), that the value is in the line, and maintains one value per entity and attribute over time.
- **recall** (no LLM, no writes): hybrid search with a bounded recency boost; the user's current states always shown; superseded sentences marked `[outdated]`.

## 5. Key decisions and the history behind them (read before changing things)

| Decision | Why (what went wrong otherwise) |
|---|---|
| **v2: episodes are the memory; only states and events are structured** | v1 extracted a general triple graph. On Qwen2.5-3B the triples were often wrong (`user allergic_to Asha`), yet answers were right because the source sentence came along. Every repair (quotes, holder labels, learned cardinality, role aliases) added fields for a 3B model to get wrong and doubled sleep time (~40 s to ~110 s for 3 turns). Facts found swung 6/12 → 0/12 → 5/12 between patches. |
| Records are never deleted | v1 pruned raw turns after 7 days and kept the lossy extraction forever, so a bad extraction was permanent. Now `rebuild` can re-read everything with a better model. |
| The model cites line numbers, not quotes | Copying quotes made the answer long (the slow part) and needed fuzzy matching. A line number is a few tokens and exact. |
| Only one-valued states are structured | One-valued state is what plain search cannot do (what is true *now*). Many-valued facts (likes, allergies, hobbies) are found as sentences; no cardinality guessing is needed. |
| Who a memory is about is resolved in code; the model's label is not trusted | Qwen2.5-3B labelled "I live in Pune" as about someone else. First-person words in the user's own lines mean the user; "she" carries over the last name before it (Panini's anuvrtti). The two pronoun lists are config, replaceable per language. |
| No hand-written meaning heuristics | The pre-v1 engine was full of them and broke on ordinary input. Structural rules (sentence ends, code fences, dates, closed pronoun classes) are fine. |
| Recall never writes and never calls an LLM | Speed, and "retrieval is not evidence": a memory must not get stronger because it was retrieved. |
| Similarity weighted against a data-adaptive floor | bge-small gives unrelated short texts about 0.4–0.6 cosine, so plain rank fusion pulled in junk. |
| Others' states need lexical evidence to be shown | In a fresh space (before the floor warms up) vector-only matches put unrelated people's states into "Now true". |
| Recency is a bounded boost (`use_weight`) | With ACT-R decay 0.5 alone, a 90-day-old memory scores about 0.1 of a fresh one, so fresh trivia would outrank an old relevant fact. The boost moves rank by at most 30%. |
| Dates: the model copies date words, code resolves them in the user's timezone | Small models get date arithmetic wrong; UTC log dates put "yesterday" on the wrong day between 00:00 and 05:30 IST. |
| The novelty gate drops only near-copies (0.97) | Embeddings barely see negation: "is allergic to" vs "is not allergic to" is 0.883, so a looser gate dropped corrections. |
| Today's date in the memory block | Otherwise the model said "a few months ago" for an event four days earlier. |
| No eval names in the extraction prompt | v1 put "my sister Asha" in the prompt while the eval tested that exact sentence. |

## 6. What has been verified

- **Tests:** `python -m pytest -q`, all passing, including the proxy against a fake upstream (streaming, disconnects, malformed bodies, CSRF guard, secret masking).
- **Real model:** Qwen2.5-3B-Instruct Q4_K_M on llama.cpp, eval dev split; see §9 for how to run it. Results of the latest run are in `tasks.md`.
- **Not verified:** any public benchmark (LongMemEval); the held-out test split beyond a single run; client configs in `docs/CLIENTS.md`; runtimes other than Ollama and llama.cpp.

## 7. Known issues and limitations (most important first)

1. **Extraction coverage is bounded by the sleep model.** A 3B model misses states. Sentences are still found by search, but "Now true" and `[outdated]` depend on sleep. `rebuild` with a 7B model is the cheapest improvement.
2. **The eval is small** (8 dev cases, 10 test cases, keyword-scored answers). Treat differences of one or two cases as noise.
3. **English-centred perception.** The prompt, date parser, FTS stemmer and pronoun lists are English; Hinglish works only as far as the model and embedder manage.
4. **"user" filed for someone else's state is not caught** when the line has first-person words ("my sister Asha lives in Delhi" filed as the user's). The sentence shown with it lets the chat model correct it.
5. **Fresh spaces are noisier** until `novelty_warmup` records exist.
6. **Token counts are an estimate** (`chars_per_token` = 3.5).
7. **Sleep competes with chat** for the GPU on small machines; `--sleep-upstream` can point sleep at another server.
8. **No `LICENSE` file yet,** although `pyproject.toml` says Apache-2.0.

## 8. What to do next

1. Run the held-out test split once and report it next to dev, with the episodic-only baseline (`--set enable_semantic=false`).
2. LongMemEval (oracle split) with last-k, full-context and plain-RAG baselines.
3. A stronger sleep model through `rebuild` (7B on a second server) and measure the gain.
4. Grow the eval to 30–50 cases, judged by a larger model instead of keywords.
5. Coding sessions: tool-output memory (what was tried, what worked), now as a separate, tested module.

## 9. Working on the code

- **Tests first.** The fakes make tests deterministic and fast. `seed_state` creates a state backed by a real sentence.
- **Every mechanism gets an `enable_*` switch** and stays only if turning it off hurts the eval.
- **Run the eval after every change to sleep or recall:** `python -m eval.run --upstream URL --model MODEL --label NAME` (dev). Ablate with `--set enable_x=false`. Compare with `--compare eval/runs/a.json eval/runs/b.json`. Run `--split test` only to report, never to tune.
- **Debug extraction by replaying:** `smruti.sleep.extract.build_messages(lines, context)` with `LLMClient.chat_json(..., schema=JSON_SCHEMA)`. Import the module with `importlib.import_module("smruti.sleep.extract")`, because `smruti.sleep` re-exports a function named `extract`.

## 10. History

- Pre-v1: a coding-agent memory with anti-memories, MCP tools and hooks; replaced.
- v1 (Plans 1–2, Batches A and C): proxy, triple graph with spreading activation, priming, summaries, procedures, quotes, kartā labels, learned cardinality. Superseded by v2 for the reasons in §5; it is in git history.
- v2: the general, episode-first layer described here.
