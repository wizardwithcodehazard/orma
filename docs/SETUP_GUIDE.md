# Smruti Setup & Installation Guide

This guide provides instructions for setting up **Smruti** across different operating systems (macOS, Linux, Windows), configuring virtual environments, and resolving system dependencies.

---

## 1. Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12 (Check via `python --version` or `python3 --version`).
- **RAM**: Minimum 500 MB free memory (Smruti runs Quantized `bge-small-en-v1.5` embeddings on CPU via ONNX Runtime).
- **Disk Space**: ~150 MB for the local embedding model cache.
- **Internet Access**: Required once on first run to download the local model weights (~130 MB); completely offline afterward.

---

## 2. OS-Specific Installation

### macOS (Apple Silicon M1/M2/M3 & Intel)

1. **Verify Python & Homebrew**:
   ```bash
   brew install python@3.11
   ```

2. **Create a Dedicated Virtual Environment**:
   ```bash
   python3 -m venv ~/.smruti-env
   source ~/.smruti-env/bin/activate
   ```

3. **Install Smruti**:
   ```bash
   # From source repository:
   git clone https://github.com/wizardwithcodehazard/smriti.git
   cd smriti
   pip install -e .
   ```

4. **Verify CLI**:
   ```bash
   smruti --help
   ```

---

### Linux (Ubuntu, Debian, Fedora, Arch)

1. **Install Python Build Tools**:
   ```bash
   # Ubuntu / Debian
   sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
   
   # Fedora
   sudo dnf install -y python3 python3-pip git
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv ~/.smruti-env
   source ~/.smruti-env/bin/activate
   ```

3. **Install Smruti**:
   ```bash
   git clone https://github.com/wizardwithcodehazard/smriti.git
   cd smriti
   pip install -e .
   ```

4. **Verify CLI**:
   ```bash
   smruti status
   ```

---

### Windows (PowerShell / Command Prompt)

1. **Ensure Python is on PATH**:
   Download Python 3.11 or 3.12 from [python.org](https://www.python.org/) and check **"Add Python to PATH"** during installation.

2. **Open PowerShell as User**:
   ```powershell
   # Create a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If execution policy errors occur, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

3. **Install Smruti**:
   ```powershell
   git clone https://github.com/wizardwithcodehazard/smriti.git
   cd smriti
   pip install -e .
   ```

4. **Verify Installation**:
   ```powershell
   smruti status
   ```

---

## 3. Storage Backends

By default, Smruti creates a local SQLite database in `.smruti/smruti.db` inside your active project root.

### Pluggable PostgreSQL Backend (Optional)
For team environments or centralized services, configure PostgreSQL by setting the environment variable:

```bash
export SMRUTI_DATABASE_URL="postgresql://user:password@localhost:5432/smruti_db"
```
Or in Windows PowerShell:
```powershell
$env:SMRUTI_DATABASE_URL="postgresql://user:password@localhost:5432/smruti_db"
```

Smruti automatically applies migrations and provisions tables on startup.

---

## 4. Verifying Model Embeddings

On first run, `fastembed` downloads the quantized model `bge-small-en-v1.5`. You can verify this works offline by running:

```bash
smruti remember "Strict test verification fact" --category "test"
smruti recall "test verification"
```

If the heuristic is returned with a strength score > 1.0, your local embeddings and storage pipeline are fully operational.

---

## 5. Troubleshooting & FAQs

### How do I unblock a false positive anti-memory?
If an anti-memory is overly aggressive and blocks a legitimate action:
1. List active anti-memories:
   ```bash
   smruti status
   ```
2. Deactivate the specific pattern using the Python SDK or MCP `smruti_forget`:
   ```python
   from smruti import SmrutiMemory
   mem = SmrutiMemory()
   mem.inhibitory.deactivate_anti_memory("the_signature_name")
   mem.close()
   ```

### How do I reset or wipe the local project memory?
Simply remove the local `.smruti` directory:
- **macOS / Linux**: `rm -rf .smruti`
- **Windows**: `Remove-Item -Recurse -Force .smruti`
Running `smruti init` will re-provision a clean, empty database.

### What if I see "database is locked" errors during heavy parallel runs?
Smruti sets `PRAGMA busy_timeout = 5000` by default, giving processes 5 seconds to wait for WAL transactions. If you are running dozens of concurrent subagents, either increase `busy_timeout` or switch to the PostgreSQL backend via `SMRUTI_DATABASE_URL`.

### Migrating from Cognee or Mem0
If you are moving from Cognee or Mem0:
1. Export your existing facts to a JSON file with `rule_text` and `category` fields.
2. Ingest them directly using the Smruti SDK:
   ```python
   import json
   from smruti import SmrutiMemory

   mem = SmrutiMemory()
   with open("exported_facts.json") as f:
       facts = json.load(f)
   for item in facts:
       mem.remember(item["rule_text"], category=item.get("category", "general"))
   mem.close()
   ```

