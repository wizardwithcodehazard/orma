# Implementation status

Plans 1 (memory core) and 2 (proxy and CLI) are implemented, plus the accuracy and human-memory work listed in [tasks.md](tasks.md). Handover notes: [HANDOVER.md](HANDOVER.md). See [tasks.md](tasks.md) for what is next and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how it fits together.

## Module map

| File | Responsibility |
| --- | --- |
| `smruti/config.py` | `SmrutiConfig`: every tunable and the ablation switches; `SMRUTI_DIR`. |
| `smruti/embed.py` | Embedder protocol, `FastEmbedder` (bge-small via fastembed, lazy). |
| `smruti/messages.py` | `message_text`: text of an OpenAI-style message. |
| `smruti/llm.py` | `LLMClient` for OpenAI-compatible chat, JSON mode with one retry. |
| `smruti/store/db.py` | SQLite connection, schema, vec and FTS tables. |
| `smruti/store/records.py` | Record dataclasses and fact/procedure text rendering. |
| `smruti/store/index.py` | Vector and lexical search helpers. |
| `smruti/store/store.py` | `Store`: reads, writes, leases, novelty stats, pruning, stats. |
| `smruti/encode.py` | `Encoder`: per-turn encoding with the adaptive novelty gate. |
| `smruti/recall/` | `Recaller` (`__init__.py`): search fusion, priming, ranking; `link.py` entity linking; `activation.py` spreading activation with the fan effect; `rank.py` fusion, base-level activation, importance; `pack.py` budgeted rendering. |
| `smruti/sleep/` | `Sleeper` (`__init__.py`), LLM extraction and schema (`extract.py`), replay voting (`vote.py`), grounding and symbolic consolidation (`consolidate.py`). |
| `scripts/demo_ollama.py` | End-to-end demo against a real model, with and without memory. |
| `scripts/latency_smoke.py` | Latency and hub-robustness check with real embeddings, no LLM. |
| `smruti/memory.py` | `Memory`, the SDK entry point. |
| `smruti/serve/proxy.py` | OpenAI-compatible proxy (Starlette). |
| `smruti/serve/capture.py` | Which turns and replies to record from proxied traffic. |
| `smruti/serve/scheduler.py` | Background sleep thread. |
| `smruti/cli.py` | `smruti` command (Typer). |

## Public API

- `smruti.Memory(path=None, upstream=None, model=None, config=None, embedder=None, llm=None)` with `build_context`, `observe`, `chat`, `sleep`, `due_spaces`, `stats`, `facts`, `forget`, `close`.
- `smruti.SmrutiConfig`.
- CLI: `smruti proxy | sleep | facts | recall | forget | stats`.
- HTTP: `POST /v1/chat/completions` with memory; all other `/v1/*` passed through. Space from `X-Smruti-Space` or the body's `user`.

## Tests

`python -m pytest -q`: 248 passed. Tests use a deterministic fake embedder and fake LLM, so they need no model download.

## Deviations from the spec

- The spec's MCP and hook interfaces are deferred. The only integrations today are the proxy, CLI and Python SDK.
- Token counting is an estimate (`chars_per_token`), not the upstream tokenizer.

## Known limitations

- Weak lexical matches can still pull loosely related facts into the prompt.
- Trimming history does not yet keep tool-call and tool-result pairs together.
- No benchmark results yet (Plan 3).
- Extraction quality is bounded by the sleep model: with `llama3.2:3b` some facts are misattributed or missed (see HANDOVER.md).
