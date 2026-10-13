# Smruti handover

Everything you need to take over Smruti: what it is, how to run it, how it is built and why, what has been verified, what is broken, and what to do next.

## 1. What it is, in one paragraph

Smruti gives small local LLMs (reference: `llama3.2:3b` via Ollama, 4k-token budget, 8 GB laptop) long-term memory. It runs as an OpenAI-compatible proxy: clients such as OpenCode, Open WebUI or any OpenAI SDK point at `http://localhost:8080/v1`. Smruti then:

- merges the relevant memories into each request, under a fixed token budget, without calling an LLM;
- records the conversation;
- in the background, while the user is idle, uses the same local model to turn conversations into a structured memory.

That memory is a graph of entities and facts with time validity, plus dated events, lessons from tool use, and topic summaries. The design principle is "neural for perception, symbolic for memory": the LLM only reads text into a fixed schema, and code decides identity, contradiction, evidence and ranking.

## 2. Run it (about 5 minutes)

Python 3.10 or newer (macOS system Python 3.9 does not work). Use Homebrew Python or `uv`.

```bash
git clone https://github.com/meluhhan/smruti.git && cd smruti
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q                      # 248 pass, no downloads needed

ollama pull llama3.2:3b
smruti proxy --model llama3.2:3b         # terminal 1
```

```bash
# terminal 2: scripted end-to-end demo against the real model, with and without memory
export SMRUTI_DIR=/tmp/smruti-demo && rm -rf $SMRUTI_DIR   # restart the proxy with the same SMRUTI_DIR
python scripts/demo_ollama.py

python scripts/latency_smoke.py          # recall/encode latency with real embeddings, no LLM
smruti facts --history                   # what is remembered (including facts that are no longer true)
smruti recall "any message"              # the memory block that would be injected
```

The first run downloads the bge-small embedding model (about 130 MB). Memory lives in `~/.smruti/memory.db` unless `SMRUTI_DIR` is set. Reset it by deleting the directory.

## 3. Repo map

| Path | What it is |
|---|---|
| `smruti/config.py` | `SmrutiConfig`: every tunable and every ablation switch (`enable_*`) |
| `smruti/store/` | SQLite schema plus migrations (`db.py`), vector (sqlite-vec) and FTS5 index helpers (`index.py`), records, `Store` |
| `smruti/encode.py` | Per-turn encoding with an adaptive novelty gate |
| `smruti/recall/` | The request path: search fusion, entity linking, spreading activation (`activation.py`), base-level activation and fusion (`rank.py`), budgeted packing (`pack.py`), priming |
| `smruti/sleep/` | Background consolidation: LLM extraction with a JSON schema (`extract.py`), replay voting (`vote.py`), grounding and symbolic consolidation (`consolidate.py`), batching, retries and the lease (`__init__.py`) |
| `smruti/serve/` | The proxy (Starlette), turn capture from traffic, the background sleep scheduler |
| `smruti/memory.py`, `smruti/cli.py` | Python SDK (`Memory`) and the `smruti` command |
| `tests/` | 248 tests; `conftest.py` has `FakeEmbedder`, `NoisyEmbedder` and `FakeLLM` |
| `scripts/` | Real-model demo and latency smoke test |
| `docs/ARCHITECTURE.md` | Engineering overview |
| `docs/CLIENTS.md`, `docs/SETUP_GUIDE.md` | Client setup and troubleshooting |
| `docs/superpowers/specs/…design.md` | The design spec, the source of truth for rationale. Keep it updated when behaviour changes |
| `docs/superpowers/plans/` | The implementation plans that built Plans 1 and 2 (historical; code has since moved on) |

## 4. How a request and a sleep work

**Request (no LLM, no DB writes, about 26 ms with 2k facts):**

1. The query is linked to known entities: by embedding similarity weighted against the space's own noise floor, and by exact name containment.
2. Search hits and primed entities from recent requests become cues.
3. Activation spreads along current facts and events, divided by each entity's fan (ACT-R fan effect, so a hub like `user` does not flood the prompt).
4. Each fact is scored as relevance × base-level activation (ACT-R power-law decay over its evidence times) × importance (surprise) × confidence.
5. Results are packed into the budget as a `## Memory` block: core facts, facts, past events, lessons, background, recent notes. Today's date is included.

**Sleep (background, when 20 turns are pending or after 120 s idle):**

1. Pending turns are batched to fit `sleep_prompt_tokens`.
2. The LLM extracts `{user_name, entities, facts, events, procedures, summary_updates}`, using JSON-schema-constrained decoding where the runtime supports it.
3. The grounding filter drops anything whose words are not in the batch's own log.
4. Symbolic consolidation merges entities, normalizes predicates (with cardinality), closes contradicted one-valued facts (`valid_to`), handles back-dated facts only when both dates were stated, adds evidence (idempotent per batch), stores events and lessons, and updates summaries.
5. Only one sleeper runs per space at a time (lease). A failed batch is retried, then retired after 3 attempts.

## 5. Key decisions and the history behind them (read before changing things)

| Decision | Why (what went wrong otherwise) |
|---|---|
| No hand-written natural-language heuristics (no regex or keyword lists over meaning) | The old engine (pre-v1, still on the `main` and `feature/tier2-cognitive-enhancements` branches of the original repo) was full of them, and they blocked `python --version` after `python train.py` failed. Structural parsing (dates, JSON, secret formats) is fine. |
| Recall never writes and never calls an LLM | Speed, and "retrieval is not evidence". Rules must not get stronger just because they were retrieved. |
| Spreading activation with the fan effect instead of hop distance | With hop distance, 2,000 trivia facts about `user` drowned "Asha is allergic to peanuts" (found by `latency_smoke.py`). |
| Similarity weighted against a data-adaptive floor | bge-small gives unrelated short texts about 0.4–0.6 cosine, so rank-only fusion pulled in junk. |
| Grounding filter | The 3B model copied "lives in Pune" from the existing summary in its prompt, which overrode the Bangalore move. |
| Votes add confidence and never gate admission (default `sleep_votes=1`) | Gating on agreement made real runs worse: the noisy second pass dropped true facts. |
| Single `user_name` field plus alias guards | With a `user_names` list, llama3.2:3b put the sister's name in it in 3 out of 3 runs, so her allergy became the user's. |
| Stated vs inferred time; same-day time counts as not stated | Without this, "I moved to Bangalore last month" was stored as history and Pune stayed current. |
| Events are "real-life happenings, never the conversation" | Otherwise the model logged "assistant asks about user's work" as events. |
| Today's date in the memory block | Otherwise the model said "a few months ago" for an event four days earlier. |
| No MCP or agent hooks in v1 | The proxy already sees tool calls and results in the conversation. |

## 6. What has been verified

- **Unit and integration tests:** 248, including the proxy against a fake upstream (streaming, disconnects, malformed bodies, CSRF guard, secret masking).
- **Real model:** `llama3.2:3b` on an M1 8 GB via `scripts/demo_ollama.py`.

| Check | Result |
|---|---|
| The move: Pune to Bangalore | correct: Pune is closed, Bangalore is current |
| "Which city should I search for weekend cafes?" | "You're in Bangalore…" (plain Ollama has to ask) |
| "When did I last go trekking, and who came with me?" | "Sinhagad fort with your sister Asha on 2026-10-09" (plain Ollama: "I can't know") |
| "Cookies for my sister?" | **flaky**: correct (peanut-free) when "user has sister Asha" was extracted, misses the allergy when it wasn't |
| Sleep time | about 10–25 s per batch on the M1; recall < 1 s |

- **Not verified:**
  - any benchmark;
  - the client config examples in `docs/CLIENTS.md` (OpenCode, Continue), which were never tried against the real apps;
  - runtimes other than Ollama.

## 7. Known issues and limitations (most important first)

1. **Extraction quality is bounded by the sleep model.** With llama3.2:3b, facts are sometimes misattributed ("user is allergic" instead of the sister), links get missed ("has sister Asha"), and relative times are left unresolved. This is the main weakness. Plans are in §8.
2. **Cardinality guesses stick.** A predicate's one/many is decided by the first extraction and never revised.
3. **The grounding filter is literal.** Paraphrases ("mom" in the chat, "mother" in the extraction) get dropped. Watch `facts_ungrounded` in sleep reports.
4. **Guard (b) on aliases** rejects a user name that already exists as another entity from an earlier batch.
5. **Fresh spaces are noisy.** Until a space has enough history (`novelty_warmup` = 50 turns), the similarity floor is off and recall is noisier.
6. **Token counts are an estimate** (`chars_per_token` = 3.5), not the model's tokenizer.
7. **Trimming history can split** a tool call from its tool result.
8. **Sleep competes with chat** for the GPU on small machines.
9. **A Starlette deprecation warning** in tests (`httpx` with `TestClient`); harmless for now.
10. **No `LICENSE` file yet,** although `pyproject.toml` says Apache-2.0. Add the license text before announcing.

## 8. What to do next (in this order; also in `tasks.md`)

1. **Deterministic dates.** The model copies the phrase ("last Saturday"), and a date library resolves it against the turn's timestamp. Small models should not do date arithmetic.
2. **Show source sentences.** Store each fact with its source sentence and show the sentence in the prompt, so a misattributed triple cannot mislead the answer. Probably the biggest quick win.
3. **A small labelled eval set.** 30–50 conversations targeting the failure types in §7.1, run after every change. Decide by numbers, not by eye. (Voting looked good in theory and was worse in practice.)
4. **LongMemEval (Plan 3).** Use the oracle split first (500 questions, about 7k tokens of history each, about 1.5 min per question on an M1). It's free on a Kaggle GPU notebook with Ollama inside, judged by Gemini through its OpenAI-compatible endpoint (rate limits vary; cache verdicts so runs can resume).
   - Baselines: last-k, full-context, naive RAG.
   - Then ablate every `enable_*` switch.
   - The dataset is `xiaowu0162/longmemeval-cleaned` on Hugging Face.
5. **Rewrite before extracting** (resolve pronouns into standalone sentences), and a yes/no verification pass per extracted fact.
6. **A fixed relation vocabulary** as a schema enum, so cardinality per relation is deterministic.
7. **`--sleep-model`,** so sleep can use a stronger model than chat when RAM allows.
8. **Packaging.** Publish to PyPI and add `smruti up` (check Ollama, pull the model, start the proxy), so users can run `pipx install smruti && smruti up`.
9. **Later:**
   - GLiNER or coreference models as fast perception;
   - a LoRA extraction adapter trained on corrected sleep output;
   - `smruti correct` and asking the user when a memory is uncertain;
   - an optional MCP server.

## 9. Working on the code

- **Tests first.** The fakes make tests deterministic and fast. `NoisyEmbedder` mimics a real embedder's ~0.5 baseline similarity; use it when testing relevance.
- **Ablations.** Every new mechanism gets an `enable_*` switch in `SmrutiConfig`, so its effect can be measured.
- **Check against the real model.** Unit tests have repeatedly passed while the real 3B model behaved differently. After any change to sleep or recall, run `scripts/demo_ollama.py` and read the facts and answers.
- **Debug extraction by replaying.** To see what the model actually extracted, call `LLMClient.chat_json` with `build_messages(...)` and `JSON_SCHEMA` from `smruti.sleep.extract`. Import the module with `importlib.import_module("smruti.sleep.extract")`, because `smruti.sleep` re-exports a function named `extract` that shadows the module.
- **Keep the spec current.** Update the design spec and `docs/ARCHITECTURE.md` when behaviour changes.

## 10. History

- The project started as a coding-agent memory (anti-memories for failing shell commands, MCP tools, hooks). That engine was replaced by this design.
- The concurrency work from the `Index` branch (write retries, queue worker) was merged into the old engine first; the rewrite superseded it.
- The original repository is `wizardwithcodehazard/smruti`. This repository (`meluhhan/smruti`) continues from the `feature/local-memory` branch, which is now `main`.
