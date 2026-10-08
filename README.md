# Smruti (स्मृति)

> **Biologically Inspired, Dual-Valence Cognitive Memory Engine for Autonomous AI Agents**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Protocol_Compliant-green.svg)](https://modelcontextprotocol.io/)
[![Zero External DB](https://img.shields.io/badge/Vector_DB-Zero_Server_In--Process-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)]()

---

## Why Smruti?

Most AI agent memory frameworks focus solely on **positive RAG**: storing text documents or knowledge graph entities. 

However, when coding agents operate autonomously, their most frequent failure modes are:
1. **Loop Amnesia**: Repeating the same broken terminal command, obsolete build flag, or failing compiler fix 5 times in a row.
2. **Catastrophic Shell Actions**: Accidentally running `rm -rf`, modifying protected branches, or wiping production tables.
3. **High Latency Overhead**: Calling an external LLM to check safety before every single tool action adds 1–3 seconds of delay per turn.

**Smruti solves this by mirroring biological neuroscience:**
- **Tier 1 (Episodic Buffer)**: Append-only SQLite WAL stream. Sub-millisecond execution logging with zero LLM overhead.
- **Tier 2 (Sleep Consolidation)**: Offline background distillation that extracts rules from recurring patterns and prunes verbatim noise.
- **Tier 3A (Cortical Positive Mesh)**: Positive rules and preferences with mathematical Ebbinghaus forgetting curves and Hebbian reinforcement.
- **Tier 3B (Inhibitory Reflex Mesh)**: Fast preflight gate (< 2ms) that intercepts dangerous actions and known dead ends before execution.

---

## Architecture Overview

```text
[ Agent Action Loop ]
         │
         ▼
┌─────────────────────────────────┐
│ Tier 3B: Inhibitory Gate        │ < 2ms (Exact substring, dense vector, guarded regex)
│ (Catastrophic Action Intercept) │
└───────────────┬─────────────────┘
                │ Passed
                ▼
┌─────────────────────────────────┐
│ Execution & Tier 1 Buffer       │ < 1ms SQLite WAL append (Zero LLM overhead)
│ (Sub-millisecond Episode Log)   │
└───────────────┬─────────────────┘
                │ Periodic / Task Completion
                ▼
┌─────────────────────────────────┐
│ Tier 2: The Sleep Cycle         │ Offline background distillation & Ebbinghaus decay
│ (Consolidation Engine)          │
└───────────────┬─────────────────┘
                │ Distills & Promotes
                ▼
┌─────────────────────────────────┐
│ Tier 3A: Cortical Knowledge     │ Vectorized BLAS cosine search + Graph spreading
│ (Positive Heuristics & Rules)   │
└─────────────────────────────────┘
```

For detailed mathematical specifications and engineering internals, read the [Technical Architecture Documentation](file:///c:/Users/Sahil/Desktop/cognee/smruti/docs/ARCHITECTURE.md).

---

## Quickstart

### 1. Installation

Smruti runs 100% locally with zero external API keys required. It uses in-process FastEmbed (`bge-small-en-v1.5`) via ONNX Runtime on CPU.

#### macOS & Linux
```bash
# Clone the repository
git clone https://github.com/wizardwithcodehazard/smruti.git
cd smruti

# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

#### Windows (PowerShell)
```powershell
git clone https://github.com/wizardwithcodehazard/smruti.git
cd smruti

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e .
```

*For complete platform troubleshooting, see the [Multi-OS Setup Guide](file:///c:/Users/Sahil/Desktop/cognee/smruti/docs/SETUP_GUIDE.md).*

---

### 2. CLI Inspection & Commands

Smruti ships with a rich developer CLI:

```bash
# Initialize Smruti database in active project (.smruti/smruti.db)
smruti init

# View status of active cortical rules, anti-memories, and episodic buffer
smruti status

# Store a declarative invariant or preference directly
smruti remember "Never deploy to production on Friday afternoon without approval" --category "devops"

# Ingest an entire architecture document or guide into chunked memory
smruti ingest docs/ARCHITECTURE.md --category "architecture"

# View formatted memory context ready for prompt injection
smruti context

# Recall top-scoring heuristics using dense vector search
smruti recall "deployment approval"

# Record an inhibitory anti-memory (block catastrophic commands)
smruti anti-fact "Do not suggest Redux Toolkit; strictly use Zustand"

# Run offline Tier 2 Sleep consolidation cycle
smruti sleep

# Export memory state into a portable JSON bundle for team sharing
smruti export --output smruti_bundle.json

# Import a bundle into another machine or workspace
smruti import smruti_bundle.json
```

---

## AI Agent Integration (MCP)

Smruti provides native support for the **Model Context Protocol (MCP)**, connecting directly to **Cursor, Claude Desktop, Google Antigravity, and Windsurf**.

### Claude Desktop Configuration
Add Smruti to your Claude Desktop config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "smruti": {
      "command": "smruti",
      "args": ["mcp"]
    }
  }
}
```

### Cursor Configuration
1. Open Cursor **Settings** -> **Features** -> **MCP Servers**.
2. Click **Add New MCP Server**:
   - **Name**: `smruti`
   - **Type**: `command`
   - **Command**: `smruti mcp`

In your repository's `.cursorrules`, add:
```markdown
# Smruti Autonomous Memory Protocol
- At conversation startup, query smruti_recall_heuristics for past rules.
- When learning user constraints or project conventions, call smruti_observe or smruti_remember.
- Before running shell commands or major modifications, call smruti_preflight_check.
- After running terminal commands, log outcomes via smruti_record_episode.
- At task completion, trigger smruti_trigger_sleep.
```

### Live MCP Resource (`smruti://active-context`)
Smruti exposes a live dynamic resource URI:
- **URI**: `smruti://active-context`
- **MIME**: `text/markdown`

Subscribing agents automatically receive project directives, active preferences, and strictly forbidden anti-patterns at session start without burning tool invocation turns.

*For complete IDE and agent setup instructions, see the [Agent Integrations Guide](file:///c:/Users/Sahil/Desktop/cognee/smruti/docs/AGENT_INTEGRATIONS.md).*

---

## Python SDK Integration

You can integrate Smruti directly into your own agent loop, LangChain runnable, or custom script:

```python
from smruti import SmrutiMemory

# Initialize workspace memory
mem = SmrutiMemory(project_root="./my_project")

# 1. Store declarative facts or observe natural user conversation
mem.observe("We strictly use PostgreSQL with connection pooling as our primary database.")
mem.remember("Always format code with ruff before committing", category="code_convention")

# 2. Preflight Safety Gate: Check actions before executing (< 2ms)
check = mem.check_action("git push origin main --force")
if not check.passed:
    print(f"BLOCKED: {check.reason}")
    print(f"Suggested alternative: {check.suggested_fix}")
else:
    # Execute tool...
    mem.record_episode("git push origin main --force", outcome="Success", status="success")

# 3. Retrieve formatted context for prompt injection
prompt_context = mem.get_prompt_context()

# 4. Trigger Sleep Consolidation at milestone boundaries
report = mem.sleep()
print("Consolidation summary:", report)

mem.close()
```

---

## Team Sync & Portability

Share project reflexes and conventions with your entire team via Git:

```bash
# Export active rules and anti-memories to JSON
smruti export -o smruti_bundle.json

# Commit to version control
git add smruti_bundle.json && git commit -m "chore: sync project memory bundle"
```

Teammates and CI/CD agents can import the bundle instantly:
```bash
smruti import smruti_bundle.json
```

---

## Documentation Index

- [Multi-OS Setup & Troubleshooting Guide](file:///c:/Users/Sahil/Desktop/cognee/smruti/docs/SETUP_GUIDE.md)
- [Agent & IDE Integration Guide (Cursor, Claude, Antigravity, Windsurf)](file:///c:/Users/Sahil/Desktop/cognee/smruti/docs/AGENT_INTEGRATIONS.md)
- [Technical Architecture & Neuroscience Deep Dive](file:///c:/Users/Sahil/Desktop/cognee/smruti/docs/ARCHITECTURE.md)

---

## License

Smruti is open source under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0).
