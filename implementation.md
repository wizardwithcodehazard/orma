# Implementation status

Branch `feature/local-memory`. Plans 1 (memory core) and 2 (proxy and CLI) are implemented. See [tasks.md](tasks.md) for what is next and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how it fits together.

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
| `smruti/recall/` | `Recaller`: search fusion, entity linking, spreading activation, ranking, budgeted packing. |
| `smruti/sleep/` | `Sleeper`, LLM extraction (`extract.py`), symbolic consolidation (`consolidate.py`). |
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

`python -m pytest -q`: 150 passed. Tests use a deterministic fake embedder and fake LLM, so they need no model download.

## Deviations from the spec

- The spec's MCP and hook interfaces are deferred. The only integrations today are the proxy, CLI and Python SDK.
- Token counting is an estimate (`chars_per_token`), not the upstream tokenizer.

## Known limitations

- Weak lexical matches can still pull loosely related facts into the prompt.
- Trimming history does not yet keep tool-call and tool-result pairs together.
- No end-to-end benchmark results yet (Plan 3).
