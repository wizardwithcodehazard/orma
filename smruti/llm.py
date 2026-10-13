"""
llm.py - Minimal client for any OpenAI-compatible chat endpoint (Ollama, llama.cpp server,
LM Studio, vLLM). Sleep uses it for JSON-mode extraction; the SDK uses it for chat.
"""

import json

import httpx


class LLMError(RuntimeError):
    pass


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
            return response.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"chat request failed: {exc}") from exc

    def chat_json(self, messages: list[dict], model: str | None = None) -> dict:
        """JSON-mode chat; retries once at temperature 0 if the output is not a JSON object."""
        for temperature in (None, 0.0):
            text = self.chat(messages, model=model, json_mode=True, temperature=temperature)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise LLMError("model did not return a JSON object")

    def close(self) -> None:
        self.http.close()
