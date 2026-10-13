# Connecting clients

Start the proxy first (`smruti proxy`, see the README). It listens on `http://localhost:8080/v1` by default. Clients need no other changes: any API key value works for a local runtime unless your upstream requires one (the proxy forwards the client's headers).

The config formats of third-party clients change. The snippets below are examples; check each client's docs for the current format.

## OpenCode

An `opencode.json` custom provider using the OpenAI-compatible adapter:

```json
{
  "$schema": "https://opencode.ai/config.json",
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

## Open WebUI

Settings, Connections, OpenAI API: set the URL to `http://localhost:8080/v1` (use `http://host.docker.internal:8080/v1` if Open WebUI runs in Docker). The API key can be any non-empty string.

## Continue

In Continue's config, add a model with the `openai` provider and `apiBase` set to the proxy:

```yaml
models:
  - name: Local via Smruti
    provider: openai
    model: llama3.2:3b
    apiBase: http://localhost:8080/v1
```

## Any OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
```

## Other upstreams

Smruti talks to any OpenAI-compatible runtime. Pass its `/v1` base URL with `--upstream`:

| Runtime | Command |
| --- | --- |
| Ollama (default) | `smruti proxy` |
| LM Studio | `smruti proxy --upstream http://localhost:1234/v1` |
| llama.cpp `llama-server` | `smruti proxy --upstream http://localhost:8080/v1 --port 8081` (pick a different proxy port than the server) |
| vLLM | `smruti proxy --upstream http://localhost:8000/v1` |

The ports are each runtime's usual defaults; adjust them to your setup. Sleep uses the model named in client requests, or the one you pass with `--model`.

## Requirements and limits

- Use `--model` to pick the sleep model; otherwise the first model a client requests is used, which may be a small helper model.
- Add the header `X-Smruti-Skip: 1` to title-generation, autocomplete and other helper requests. They get no recall, are not recorded, and the header is not forwarded upstream.
- Chat requests must have `Content-Type: application/json` (other types get a 415).
- Secrets are masked before storing (API keys, bearer tokens, AWS and GitHub tokens, private keys, `password=`/`api_key=` values), and tool output is truncated to `max_tool_chars` (default 4000).
- Browser-direct clients are not supported (the proxy sends no CORS headers). Only `/v1` paths are proxied, so Ollama's native `/api` endpoints are not.

## Memory spaces

A space is an isolated memory. Use one per user or project. The proxy picks the space from, in order: the `X-Smruti-Space` header, the request body's `user` field, the `--space` option (default `default`).

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Smruti-Space: project-a" \
  -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Remember that we use Postgres."}]}'

smruti facts --space project-a
```

## Tool calls

Tool calls and their results are captured automatically. A tool result is recorded together with the call that produced it. During sleep they become procedures that are recalled as warnings or tips on similar situations. Nothing needs to be configured in the client. Streaming and non-streaming requests are both supported; other `/v1` endpoints (such as `/v1/models`) pass through unchanged.
