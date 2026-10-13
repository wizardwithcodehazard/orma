# Setup guide

## Install

Smruti needs Python 3.10 or newer (3.9 is not supported).

```bash
git clone <this repo> && cd smruti
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # ".[dev]" adds pytest and ruff; use "." for runtime only
smruti --help
```

On first use Smruti downloads the `BAAI/bge-small-en-v1.5` embedding model (about 130 MB) once and caches it.

## Run

```bash
ollama pull llama3.2:3b
smruti proxy
```

Defaults: upstream `http://localhost:11434/v1`, listen on `127.0.0.1:8080`. Point clients at `http://localhost:8080/v1`; see [CLIENTS.md](CLIENTS.md).

## Where data lives

`~/.smruti/memory.db` (SQLite, WAL mode). Set `SMRUTI_DIR` to use a different directory. Memory spaces share this one file.

## Tests

```bash
python -m pytest -q
```

Tests use fake embedders and fake LLM transports, so they need no model download or running server.

## Troubleshooting

- **502 from the proxy.** The upstream runtime is unreachable. Check that Ollama (or your runtime) is running and that `--upstream` is the right `/v1` URL.
- **Facts never appear.** Sleep needs a model. The proxy learns it from the first request that names one; otherwise pass `--model`. `smruti sleep --model llama3.2:3b` forces a run. Background sleep starts at 20 pending turns or 120 s idle; check `smruti stats` for `pending_episodes`.
- **Sleep batches failing.** A failed batch is retried on later cycles and retired after 3 attempts. Use a model that can follow a JSON schema; very small models may extract poorly.
- **`SyntaxError` or install errors on Python 3.9.** Use Python 3.10 or newer.
- **Reset memory.** Stop the proxy and delete `~/.smruti/memory.db` (and its `-wal` and `-shm` files if present).
- **Remove one wrong fact.** `smruti facts` to find its id, then `smruti forget ID`.
