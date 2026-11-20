"""
llm.py - Minimal client for any OpenAI-compatible chat endpoint (Ollama, llama.cpp server,
LM Studio, vLLM). Sleep uses it for schema-constrained JSON extraction; the SDK uses it for chat.
"""

import json
import re

import httpx

RETRY_REMINDER = {"role": "user", "content": "Return only a valid JSON object."}


# Statuses a runtime returns for a request shape it does not support
SCHEMA_UNSUPPORTED_STATUSES = {400, 422}


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP status when the upstream answered with an error


def _loads(text: str):
    """Parses JSON, falling back to fenced stripping, trailing comma cleanup, and truncated bracket repair."""
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 1. Clean markdown codeblocks
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    # 2. Extract outermost {...}
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 3. Clean trailing commas in arrays/objects
        without_trailing = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(without_trailing)
        except json.JSONDecodeError:
            pass

    # 4. Handle truncated JSON (unclosed brackets/braces from token limit)
    start = cleaned.find("{")
    if start >= 0:
        frag = cleaned[start:].rstrip(", \n\t")
        frag = re.sub(r",\s*([}\]])", r"\1", frag)
        open_curly = frag.count("{") - frag.count("}")
        open_square = frag.count("[") - frag.count("]")
        if open_square > 0:
            frag += "]" * open_square
        if open_curly > 0:
            frag += "}" * open_curly
        frag = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass

    return None


class LLMClient:
    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 600.0, transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.json_schema_supported = True  # cleared once the upstream rejects response_format json_schema
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)

    def chat(self, messages: list[dict], model: str | None = None, json_mode: bool = False,
             temperature: float | None = None, response_format: dict | None = None,
             max_tokens: int | None = None) -> str:
        body = {"model": model or self.model, "messages": messages, "stream": False}
        if response_format is not None or json_mode:
            body["response_format"] = response_format or {"type": "json_object"}
        if temperature is not None:
            body["temperature"] = temperature
        # Cap generation length so small models cannot enter infinite loop
        effective_max = max_tokens if max_tokens is not None else (2048 if (json_mode or response_format) else None)
        if effective_max is not None:
            body["max_tokens"] = effective_max
        try:
            response = self.http.post("/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"chat request failed: {exc} - {exc.response.text[:200]}",
                           status=exc.response.status_code) from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"chat request failed: {exc}") from exc
        if content is not None and not isinstance(content, str):
            raise LLMError("chat response content is not text")
        return content or ""

    def chat_json(self, messages: list[dict], model: str | None = None, schema: dict | None = None,
                  temperature: float = 0.0) -> dict:
        """JSON chat, constrained to the JSON Schema when given; retries once at temperature 0 with a
        stricter reminder if the output is not a JSON object."""
        for attempt in range(2):
            asked = messages if attempt == 0 else messages + [RETRY_REMINDER]
            data = _loads(self._json_reply(asked, model, schema, temperature if attempt == 0 else 0.0))
            if isinstance(data, dict):
                return data
        raise LLMError("model did not return a JSON object")

    def _json_reply(self, messages: list[dict], model: str | None, schema: dict | None, temperature: float) -> str:
        """Schema-constrained decoding when the runtime supports it; a 4xx to json_schema means it does not,
        so this client falls back to plain JSON mode from then on."""
        if schema is not None and self.json_schema_supported:
            response_format = {"type": "json_schema",
                               "json_schema": {"name": "memory", "schema": schema, "strict": True}}
            try:
                return self.chat(messages, model=model, temperature=temperature, response_format=response_format)
            except LLMError as exc:
                if exc.status not in SCHEMA_UNSUPPORTED_STATUSES:
                    raise  # auth, unknown model, server errors: not a json_schema problem
                self.json_schema_supported = False
        return self.chat(messages, model=model, json_mode=True, temperature=temperature)

    def close(self) -> None:
        self.http.close()
