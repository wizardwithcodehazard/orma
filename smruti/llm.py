"""
llm.py - Minimal client for any OpenAI-compatible chat endpoint (Ollama, llama.cpp server,
LM Studio, vLLM). Sleep uses it for JSON-mode extraction; the SDK uses it for chat.
"""

import json

import httpx


RETRY_REMINDER = {"role": "user", "content": "Return only a valid JSON object."}


class LLMError(RuntimeError):
    pass


def _loads(text: str):
    """Parses JSON, falling back to the outermost {...} span (fenced or chatty output)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


class LLMClient:
    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0, transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)

    def chat(self, messages: list[dict], model: str | None = None, json_mode: bool = False,
             temperature: float | None = None) -> str:
        body = {"model": model or self.model, "messages": messages, "stream": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if temperature is not None:
            body["temperature"] = temperature
        try:
            response = self.http.post("/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"chat request failed: {exc} - {exc.response.text[:200]}") from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"chat request failed: {exc}") from exc
        if content is not None and not isinstance(content, str):
            raise LLMError("chat response content is not text")
        return content or ""

    def chat_json(self, messages: list[dict], model: str | None = None) -> dict:
        """JSON-mode chat at temperature 0; retries once with a stricter reminder if the output is not a JSON object."""
        for attempt in range(2):
            asked = messages if attempt == 0 else messages + [RETRY_REMINDER]
            data = _loads(self.chat(asked, model=model, json_mode=True, temperature=0.0))
            if isinstance(data, dict):
                return data
        raise LLMError("model did not return a JSON object")

    def close(self) -> None:
        self.http.close()
