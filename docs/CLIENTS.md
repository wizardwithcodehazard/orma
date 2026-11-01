# Connecting Clients to Orma

Orma can be connected to any LLM application, developer tool, or AI coding assistant using either **Model Context Protocol (MCP)** or the **OpenAI-Compatible Proxy**.

---

## Option 1: Native MCP (Claude Code, Cursor, Windsurf, Antigravity IDE)

Orma runs natively as a Model Context Protocol server over standard `stdio`. This provides zero-latency memory across all your coding sessions.

### In Claude Code:
```bash
claude mcp add orma -- orma mcp
```

### In Cursor (`.cursor/mcp.json` or Settings > MCP):
```json
{
  "mcpServers": {
    "orma": {
      "command": "orma",
      "args": ["mcp"]
    }
  }
}
```

### In Windsurf / Antigravity IDE (`mcp.json`):
```json
{
  "mcpServers": {
    "orma": {
      "command": "orma",
      "args": ["mcp"]
    }
  }
}
```

### One-Click Install via Smithery.ai:
```bash
npx -y @smithery/cli install orma-ai --client cursor
```

### Available MCP Tools:
- **`orma_recall(query, space)`**: Returns token-budgeted, Sutra-packed active facts, anti-patterns, and past decisions (< 20ms).
- **`orma_remember(text, kind, role, space)`**: Permanently records a project decision, architecture rule, or user constraint. Set `kind="anti_pattern"` to mark a forbidden mistake to avoid.
- **`orma_forbid(rule, space)`**: Directly registers a hard anti-pattern or prohibited behavior (e.g. "Never chain PowerShell commands with semicolons").
- **`orma_facts(space, include_history)`**: Inspects all active canonical truths without running a search.
- **`orma_forget(state_id, space)`**: Closes an obsolete or superseded constraint.

---

## Option 2: OpenAI-Compatible Proxy Mode

Start the Orma memory proxy first:
```bash
# With Ollama
orma proxy --model llama3.2:3b

# With llama.cpp
orma proxy --upstream http://127.0.0.1:9931 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"

# With one-command browser launcher
orma up --upstream http://127.0.0.1:9931 --model "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
```

The proxy listens on `http://localhost:8080/v1` by default. Point any OpenAI-compatible client to this address:

### With the Official OpenAI Python Client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "What is my job and where do I live?"}]
)
print(response.choices[0].message.content)
```

### With LangChain:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",
    model="llama3.2:3b"
)
```

### With Open WebUI:
1. Open **Settings > Connections > OpenAI API**.
2. Set the URL to `http://localhost:8080/v1` (or `http://host.docker.internal:8080/v1` if Open WebUI runs in Docker).
3. The API key can be any string (e.g., `none`).

### With Continue (`config.yaml`):
```yaml
models:
  - name: Local via Orma
    provider: openai
    model: llama3.2:3b
    apiBase: http://localhost:8080/v1
```

---

## Upstream Runtime Compatibility

Orma speaks to any OpenAI-compatible runtime:

| Runtime | Upstream URL Flag | Default Upstream Port |
|---|---|---|
| **Ollama** | `--upstream http://localhost:11434/v1` | `11434` |
| **llama.cpp (`llama-server`)** | `--upstream http://127.0.0.1:9931/v1` | `9931` (or custom) |
| **LM Studio** | `--upstream http://localhost:1234/v1` | `1234` |
| **vLLM** | `--upstream http://localhost:8000/v1` | `8000` |
