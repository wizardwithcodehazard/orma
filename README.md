# Smruti (स्मृति)

> **Learned failure memory for coding agents.** Smruti watches what your agent actually runs, learns which commands keep failing and what fixed them, and warns the agent before it walks into the same dead end again.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Protocol_Compliant-green.svg)](https://modelcontextprotocol.io/)
[![Zero External DB](https://img.shields.io/badge/Vector_DB-Zero_Server_In--Process-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-82%20passed-brightgreen.svg)]()

---

## Why Smruti?

Autonomous coding agents fail in loops: the same missing dependency, the same broken build flag, the same command retried five times. Most agent memory layers (Mem0, Zep, Letta, Cognee) store facts and preferences you tell them. Smruti instead learns from **execution traces**: what was run, what failed, and what fixed it. Everything runs locally on SQLite with in-process embeddings and no API keys.

## How it works

1. **Capture**: Claude Code hooks (or `smruti run -- <cmd>`) record every command and its outcome automatically. Nothing depends on the agent remembering to call a tool.
2. **Sleep**: consolidation groups repeated failures by *exact command + error signature* into learned warnings, and pairs failures with the related command that fixed them into rules.
3. **Preflight**: before a command runs, a matching learned failure adds a warning (reason + known fix) to the agent's context. Rules you write yourself, such as "never `git push --force` to main", block.

**Design principles**

- **Warn on what was learned, block only on what you wrote.** A wrong block costs more than a missed loop.
- **Learned memory corrects itself.** A learned warning is retired as soon as the same command succeeds, and expires after 7 days. A learned fix starts at 0.5 confidence and only strengthens when the same fix is seen again, not when it's retrieved.
- **What you tell it persists.** Only learned rules decay; facts you `remember` stay until you `forget` them.

```text
[ Agent runs a shell command ]
         │  PreToolUse hook / smruti run
         ▼
┌─────────────────────────────────┐
│ Preflight Gate                  │ learned failure → warning in agent context
│                                 │ user-written rule → block (literal / regex / semantic)
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Episodic Buffer                 │ every command + outcome, SQLite WAL, secrets masked
└───────────────┬─────────────────┘
                │ every N commands / on demand
                ▼
┌─────────────────────────────────┐
│ Sleep Consolidation             │ (command, error) clusters → learned warnings
│                                 │ failure → related fix   → tentative rules
│                                 │ success → retire warning; decay unconfirmed rules
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Rules & Preferences             │ recalled by vector + keyword search for prompt context
└─────────────────────────────────┘
```

For detailed mathematical specifications and engineering internals, read the [Technical Architecture Documentation](docs/ARCHITECTURE.md).

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

*For complete platform troubleshooting, see the [Multi-OS Setup Guide](docs/SETUP_GUIDE.md).*

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

## Automatic Capture (recommended)

### Claude Code hooks
Add to `.claude/settings.json` (project) or `~/.claude/settings.json` (global):

```json
{
  "hooks": {
    "PreToolUse":         [{"matcher": "Bash", "hooks": [{"type": "command", "command": "smruti hook"}]}],
    "PostToolUse":        [{"matcher": "Bash", "hooks": [{"type": "command", "command": "smruti hook"}]}],
    "PostToolUseFailure": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "smruti hook"}]}]
  }
}
```

- Every Bash command and its outcome is recorded.
- Learned failures show up as a warning in Claude's context, without changing your permission prompts.
- Rules you wrote that should block are denied, with the reason shown to Claude.
- The hook fails open: if smruti errors, the command still runs.

### Any agent or shell
```bash
smruti run -- npm run build   # preflight, run, record exit status
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

*For complete IDE and agent setup instructions, see the [Agent Integrations Guide](docs/AGENT_INTEGRATIONS.md).*

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

# 2. Preflight gate: check actions before executing
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

## Empirical Benchmarks

Measured using the reproducible benchmark suite (`benchmarks/benchmark_engine.py`):

### 1. Preflight Safety Gate Latencies
| Check Type | Execution Mechanism | Latency | Speedup vs Cloud LLM |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Exact Substring)** | Substring Hash Match | **6.2 microseconds** | **> 15,000x faster** |
| **Tier 2 (Neural Vector)** | FastEmbed `bge-small` on CPU | **7.37 milliseconds** | **~129x faster** |
| **Cloud LLM Guardrail Call** | Remote API Network Roundtrip | ~950 milliseconds | Baseline |

### 2. Replay Evaluation on Real Command Histories
`benchmarks/replay_eval.py` replays real Bash histories in order: Claude Code transcripts (default `~/.claude/projects`) or a JSONL trace. It runs against a throwaway database and reports:
- how often a warning was followed by a real failure (precision)
- wrong flags
- repeat failures caught before they happened
- the rules learned

```bash
python benchmarks/replay_eval.py                                   # your Claude Code history
python benchmarks/replay_eval.py benchmarks/sample_trace.jsonl --sleep-every 2
```

On the bundled sample trace, all three learned rules are the real fixes (e.g. *"When 'python train.py' fails, 'pip install torch' resolved it."*). No read-only command or bare retry is mistaken for a fix. Learning only happens during sleep, so repeat failures inside one short session are caught only if consolidation runs often enough (`--sleep-every`).

### 3. BLAS Vector Search Scalability (CPU)
Batch cosine similarity using vectorized 2D BLAS dot products across rule sets:
- **100 rules**: 0.053 ms
- **1,000 rules**: 1.053 ms
- **5,000 rules**: 5.246 ms
- **10,000 rules**: 10.984 ms
- **25,000 rules**: 22.216 ms

---

## Competitor Comparison

| Capability | Smruti | Mem0 | Cognee | Zep |
| :--- | :--- | :--- | :--- | :--- |
| **Learns failures from execution traces** | **Yes** | No | No | No |
| **Sub-10ms Preflight Gate** | **Yes (< 8ms local CPU)** | No (~800ms API) | No (Graph query) | No (API roundtrip) |
| **Zero External DB Setup** | **Yes (SQLite + FastEmbed)** | No (Postgres/Cloud) | No (Tri-Store / Kùzu) | No (Cloud/Docker) |
| **Biological Ebbinghaus Decay** | **Yes (Mathematical)** | No | No | No |
| **Secret Sanitization** | **Yes (Automatic mask)** | Manual | Manual | Cloud-side |
| **Enterprise Knowledge Graph** | Minimal (Heuristic edges) | No | **Yes (Triplets/Ontology)** | No |

---

## Privacy & Security

Smruti runs 100% offline and locally. To prevent secret leakage:
- **Automated Credential Masking**: The Tier 1 episodic buffer automatically sanitizes API keys (`sk-...`), GitHub tokens (`ghp_...`), Bearer headers, AWS keys (`AKIA...`), database connection passwords, and private keys before committing any command or output to disk.

---

## Documentation Index

- [Multi-OS Setup & Troubleshooting Guide](docs/SETUP_GUIDE.md)
- [Agent & IDE Integration Guide (Cursor, Claude, Antigravity, Windsurf)](docs/AGENT_INTEGRATIONS.md)
- [Technical Architecture & Neuroscience Deep Dive](docs/ARCHITECTURE.md)

---

## License

Smruti is open source under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0).
