# Orma Setup & Installation Guide

A developer's guide to setting up, configuring, and testing **Orma** across macOS (Apple Silicon), Linux, and Windows.

---

## 1. System Requirements

* **Python:** Version **3.10 or newer** (Python 3.9 is not supported).
* **Hardware:** Any consumer laptop or workstation with 8–16 GB of RAM.
* **Storage:** Local SQLite with WAL mode (~50 MB base footprint).
* **Network:** Zero internet connectivity required at runtime. Embedded ONNX models (`bge-small-en-v1.5` and `ms-marco-TinyBERT-L-2-v2`) download once on first run (~133 MB total) and are cached locally.

---

## 2. Installation Options

### Option A: From PyPI (Standard / Production)
```bash
pip install orma-ai
```
Or with `uv`:
```bash
uv pip install orma-ai
```

### Option B: From Source (Development / Research)
```bash
git clone https://github.com/wizardwithcodehazard/orma.git
cd orma/smruti

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Verify installation
orma --help
```

---

## 3. Running Verification & Benchmarks

### Running the Test Suite (226 Tests)
```bash
python -m pytest tests/ -v
```
All 226 unit and integration tests run locally in seconds using mock embedders and mock HTTP transports, with zero external network access required.

### Reproducing the Official LoCoMo Benchmark (Zero-LLM Mode)
```bash
# Evaluates full 10-conversation dataset (1,536 QA queries) in Pure Zero-LLM Mode
python scripts/paper_eval.py --no-llm --use-palace --conversations 10 --output reports/locomo_reproduced.json
```

---

## 4. Serving Modes

Orma offers three plug-and-play serving interfaces:

### Mode 1: Model Context Protocol (MCP Server for AI Coding Assistants)
Exposes long-term memory over standard `stdio` JSON-RPC for Claude Code, Cursor, Windsurf, and Antigravity IDE:
```bash
orma mcp
```

### Mode 2: Zero-Code OpenAI Proxy (Ollama / llama.cpp / vLLM)
Transparently intercepts chat completions, retrieves memory in < 65ms, injects context into the system prompt, and logs new turns:
```bash
# With Ollama
ollama pull llama3.2:3b
orma proxy --model llama3.2:3b --upstream http://127.0.0.1:11434/v1

# With llama.cpp
orma proxy --upstream http://127.0.0.1:9931/v1 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
```

### Mode 3: One-Command Browser Web UI
```bash
orma up --upstream http://127.0.0.1:11434/v1 --model llama3.2:3b
```
Automatically opens your browser to `http://127.0.0.1:8080/#/` with live streaming chat and full memory support.

---

## 5. Where Data Lives

By default, all memories are stored in a single SQLite file:
* **Default Database Path:** `~/.orma/memory.db`

To configure a custom storage directory:
```bash
# Linux / macOS
export ORMA_DIR=/path/to/custom/dir

# Windows PowerShell
$env:ORMA_DIR = "C:\path\to\custom\dir"
```

---

## 6. CLI Administration Reference

```bash
# Ingest notes, markdown documents, or project guidelines
orma add "My sister Asha has a severe peanut allergy."
orma add docs/handbook.md

# Search memory directly across text and vectors
orma search "allergies"

# Preview the exact prompt memory block that would be injected
orma recall "What snacks should I avoid for Asha?"

# Inspect active canonical states and historical lineages
orma facts --history

# Directly register an anti-pattern or forbidden rule
orma add "FORBIDDEN RULE: Never chain terminal commands with semicolons"

# Manually trigger background sleep consolidation
orma sleep --model llama3.2:3b

# Forget a specific state
orma forget 12

# Database statistics & health check
orma stats

# Launch real-time 3D Cognitive Connectome WebGL visualizer
orma viz
```

---

## 7. Troubleshooting

* **502 Bad Gateway from Proxy:** The upstream model server is unreachable. Verify that Ollama or `llama-server` is running and accessible at `--upstream`.
* **Sleep Consolidation Fails:** Sleep requires a valid model identifier. Pass `--model` or ensure client requests include a model field. Check pending queue with `orma stats`.
* **Reset All Memory:** Stop all running Orma processes and delete `~/.orma/memory.db` (along with any `-wal` and `-shm` files).
