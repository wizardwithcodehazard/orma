# Smruti (स्मृति)

Long-term memory for small local LLMs, so a 3B model with a 4k context can remember across conversations.

Smruti runs as an OpenAI-compatible proxy in front of Ollama, llama.cpp, LM Studio or vLLM. It also ships a CLI and a Python SDK. Everything is local: one SQLite file, local embeddings, no API keys.

## Why

Small models have small context windows, so you cannot replay a long history on every request. Smruti keeps a structured memory (facts, entities, procedures, summaries) and gives the model only what matters for the current message, within a fixed token budget. The request path makes no LLM calls: recall is embeddings, SQLite and arithmetic. The model is used only in the background, to turn finished conversation into memory.

## Quickstart (Ollama)

Requires Python 3.10 or newer.

```bash
git clone <this repo> && cd smruti
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

ollama pull llama3.2:3b
smruti proxy        # upstream http://localhost:11434/v1, listening on 127.0.0.1:8080
```

Point your client at `http://localhost:8080/v1`. For example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "I live in Pune and I work on compilers."}]}'
```

After a few turns, look at what was remembered:

```bash
smruti facts
```

Sleep (turning raw turns into facts) runs in the background once 20 turns are pending, or after 120 s of idle time. To force it now: `smruti sleep --model llama3.2:3b`. The first run downloads the bge-small embedding model (about 130 MB).

## Use with your client

Any client that lets you set an OpenAI base URL works. Full setup for more clients is in [docs/CLIENTS.md](docs/CLIENTS.md).

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Where do I live?"}],
)
print(reply.choices[0].message.content)
```

Open WebUI: Settings, Connections, set the OpenAI API URL to `http://localhost:8080/v1`.

OpenCode (`opencode.json`, check OpenCode's docs for the current format):

```json
{
  "provider": {
    "smruti": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Smruti (Ollama)",
      "options": { "baseURL": "http://localhost:8080/v1" },
      "models": { "llama3.2:3b": { "name": "llama3.2:3b" } }
    }
  }
}
```

Memory spaces keep users or projects apart. A request picks its space with the `X-Smruti-Space` header, or else the request's `user` field; otherwise the proxy's `--space` default (`default`) is used.

Notes on the proxy:

- Pass `--model` to choose the model used for sleep. Otherwise the first model a client requests is adopted, which may be a small helper model.
- Send `X-Smruti-Skip: 1` on requests that should not touch memory (chat titles, autocomplete, other helper calls). They get no recall, are not recorded, and the header is not forwarded.
- Chat requests must be `application/json`; anything else gets a 415. This blocks cross-site form posts from poisoning memory.
- Secrets (API keys, bearer tokens, private keys, `password=` style values) are masked before anything is stored, and tool output is capped at `max_tool_chars` (4000).
- Only `/v1` paths are proxied; Ollama's native `/api` is not. Clients running in a browser that call the proxy directly are not supported (no CORS headers).

## How it works

```
Per request (no LLM calls)
  client -> proxy -> recall: search memory, spread activation, pack within budget
                  -> upstream model (request with memory merged into the system prompt)
                  -> reply streamed back to the client
                  -> encode: store the new user/tool turns and the reply as episodes

Sleep (background, uses the model)
  pending episodes -> LLM extracts entities, facts, procedures, summaries (schema-constrained JSON),
                      twice; facts both passes agree on go on, the rest wait as candidates
                   -> symbolic consolidation: merge entities, close contradicted facts,
                      accumulate evidence, update summaries
```

Principles:

- Neural for perception, symbolic for memory. The LLM only reads text into a fixed schema. Deciding what is the same entity, what contradicts what and what to keep is done by code over a graph.
- Facts have time validity. When a one-valued fact changes ("lives in Pune" to "lives in Mumbai"), the old fact is closed (it gets a `valid_to`), not deleted. `smruti facts --history` shows closed facts.
- Recall uses spreading activation with the fan effect. Entities mentioned in the query pass activation along their facts, divided among all of an entity's facts, so a hub like `user` gives each fact a small share and a specific entity gives its few facts a lot.
- An adaptive novelty gate drops turns that add nothing new relative to what that space has recently seen. Tool turns with outcomes are always kept.
- Sleep replays each batch: a second, sampled extraction must agree with the first before a fact enters the graph. A fact only one pass found is kept as a candidate and becomes a fact when a later conversation repeats it (or expires after 14 days).
- Only tool calls and results become procedures ("when X, doing Y fails" or "works") that are recalled as lessons next time.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Python SDK

```python
from smruti import Memory

mem = Memory(upstream="http://localhost:11434/v1", model="llama3.2:3b")

reply = mem.chat([{"role": "user", "content": "I live in Pune"}])      # recall, call the model, record both turns

mem.observe("user", "My dog is called Biscuit")                        # record a turn without calling a model
messages = mem.build_context([{"role": "user", "content": "What is my dog called?"}])  # messages to send upstream

report = mem.sleep()                                                   # consolidate pending turns
for fact in mem.facts():                                               # current facts; include_closed=True for history
    print(fact.id, fact.text)
mem.forget(fact.id)                                                    # end a fact's validity
mem.close()
```

`Memory(path=None, upstream=None, model=None, config=None, ...)`. `chat` and `sleep` need `upstream` (or an `llm`). All methods take a `space` argument (default `"default"`).

## CLI reference

| Command | What it does |
| --- | --- |
| `smruti proxy` | Run the memory proxy. Options: `--upstream`, `--host`, `--port`, `--model`, `--space`, `--sleep-poll` (seconds between background sleep checks, 0 disables). |
| `smruti sleep --model M` | Consolidate pending turns now. Options: `--upstream`, `--space`. |
| `smruti facts [ENTITY]` | List remembered facts. Options: `--space`, `--history` (include facts that are no longer true). |
| `smruti recall "text"` | Show the memory block that would be added to the prompt for that message. Options: `--space`, `--budget`. |
| `smruti forget FACT_ID` | Mark a fact as no longer true. Option: `--space`. |
| `smruti stats` | Counts of what is stored, as JSON. Option: `--space`. |

## Configuration and ablations

Settings live in `SmrutiConfig` (`smruti/config.py`); pass one to `Memory(config=...)`. The CLI uses the defaults. Memory is stored in `~/.smruti/memory.db`; set `SMRUTI_DIR` to use another directory.

| Group | Fields (defaults) |
| --- | --- |
| Budget | `budget_tokens` (4096), `chars_per_token` (3.5), `min_recent_turns` (2) |
| Sleep | `sleep_after_episodes` (20), `sleep_idle_seconds` (120), `sleep_batch_episodes` (12), `sleep_max_attempts` (3), `episode_retention_days` (7), `procedure_ttl_days` (30) |
| Replay and vote | `sleep_votes` (2; 1 disables voting), `sleep_vote_temperature` (0.7), `vote_match_threshold` (0.8), `candidate_ttl_days` (14) |
| Novelty gate | `novelty_percentile` (10), `novelty_warmup` (50), `novelty_window` (500) |
| Recall | `search_k`, `max_hops`, `hop_decay`, `recency_half_life_days`, `max_candidate_facts`, `core_min_confidence` |
| Ablation switches (all default true) | `enable_novelty_gate`, `enable_graph`, `enable_core_block`, `enable_similarity_weighting`, `enable_episode_recall`, `enable_schema_decoding` |

The `enable_*` switches turn individual components off so their effect can be measured.

## Status

Measured on an Apple Silicon laptop with bge-small embeddings and 2,000 stored facts: recall takes about 22 ms per request and encoding about 5 ms per turn. The test suite has 150 tests.

End-to-end runs with a real local model and the LongMemEval evaluation are in progress (Plan 3, see [tasks.md](tasks.md)). Known limitation: weak lexical matches can still pull loosely related facts into the prompt.

## Research

The design rationale is in [docs/superpowers/specs/2026-10-12-local-neurosymbolic-memory-design.md](docs/superpowers/specs/2026-10-12-local-neurosymbolic-memory-design.md). The engineering overview is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

License: Apache-2.0.
