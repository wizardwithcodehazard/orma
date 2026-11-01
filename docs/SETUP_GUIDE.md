# Orma Setup Guide

---

## Installation

Orma requires **Python 3.10 or newer** (Python 3.9 is not supported).

### From PyPI (Recommended):
```bash
pip install orma-ai
```

### From Source (Development):
```bash
git clone https://github.com/wizardwithcodehazard/orma.git
cd orma/smruti
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
orma --help
```

On first use, Orma downloads the `BAAI/bge-small-en-v1.5` embedding model (~130 MB ONNX) once and caches it locally.

---

## Quick Start

### 1. Run as MCP Server for Cursor / Claude Code:
```bash
orma mcp
```

### 2. Run as Proxy for Local Models:
```bash
# With Ollama
ollama pull llama3.2:3b
orma proxy --model llama3.2:3b

# With llama.cpp
orma proxy --upstream http://127.0.0.1:9931 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
```

### 3. One-Command Browser Web UI:
```bash
orma up --upstream http://127.0.0.1:9931 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
```
This automatically opens your browser to `http://127.0.0.1:8080/#/` with full long-term memory support.

---

## Where Data Lives

All memories are kept locally in **`~/.orma/memory.db`** (SQLite with Write-Ahead Logging).

- To use a custom data directory, set the `ORMA_DIR` environment variable:
  ```bash
  export ORMA_DIR=/path/to/custom/dir
  ```
- All memory spaces share this single SQLite file and are isolated by tenant/user keys.

---

## CLI Reference

```bash
# Ingest notes, files, or documents
orma add "My sister Asha is allergic to peanuts."
orma add docs/handbook.md

# Search memory directly
orma search "allergies"

# Preview the exact prompt block that would be injected
orma recall "What is Asha allergic to?"

# Inspect active facts and outdated history
orma facts --history

# Directly register an anti-pattern or forbidden rule
# (or use the orma_forbid MCP tool)
orma add "FORBIDDEN RULE: Never chain terminal commands with semicolons"

# Manually trigger background sleep consolidation
orma sleep --model llama3.2:3b

# Mark a specific obsolete state as forgotten
orma forget 12

# Database statistics
orma stats
```

---

## Running Tests

```bash
python -m pytest -q
```
All 213 unit and integration tests run locally in seconds using mock embedders and mock LLM transports, without needing external network access.

---

## Troubleshooting

- **502 Bad Gateway from Proxy:** The upstream model server is unreachable. Verify that Ollama or `llama-server` is running and accessible at `--upstream`.
- **Facts Don't Consolidate:** Sleep requires a model. Pass `--model` or ensure the client request sends a model identifier. Run `orma stats` to check `pending_episodes`.
- **Reset All Memory:** Stop any running Orma processes and delete `~/.orma/memory.db` (and any associated `-wal` or `-shm` files).
