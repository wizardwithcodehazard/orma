"""
llm.py - Extensible LLM Brain Interface for smruti.
Provides modular neural LLM arbitration during preflight and knowledge distillation during sleep cycles.
Supports OpenAI, Anthropic, Ollama, and OpenAI-compatible endpoints with zero-dependency urllib fallbacks.
"""

import json
import logging
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any

from smruti.models import AntiMemory

logger = logging.getLogger(__name__)


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    """Posts JSON payload using urllib.request (zero external dependencies)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_body = response.read().decode("utf-8")
            return json.loads(resp_body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        logger.warning("HTTP request to %s failed (%s): %s", url, e.code, err_body[:200])
        raise RuntimeError(f"LLM API request failed with status {e.code}: {err_body[:200]}") from e
    except Exception as e:
        logger.warning("Network error calling %s: %s", url, e)
        raise


class BaseLLMProvider(ABC):
    """Abstract base provider for LLM operations."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if this provider is configured and ready."""
        pass

    @abstractmethod
    def evaluate_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        pass

    @abstractmethod
    def summarize_failure(self, cluster_data: list[dict[str, Any]]) -> str:
        pass

    @abstractmethod
    def distill_resolution(self, transitions: list[dict[str, Any]]) -> str:
        pass


class HeuristicFallbackProvider(BaseLLMProvider):
    """Local, offline heuristic fallback when no LLM API is configured."""

    def is_available(self) -> bool:
        return True

    def evaluate_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        # Conservative block based on neural/exact match
        return (True, anti_memory.reason)

    def summarize_failure(self, cluster_data: list[dict[str, Any]]) -> str:
        first = cluster_data[0]
        action = first.get("action", "")[:40]
        err_snippet = first.get("result", "").strip().split("\n")[-1][:120] if first.get("result") else "Non-zero exit"
        return f"Repeated failures in '{action}'. Error: {err_snippet}"

    def distill_resolution(self, transitions: list[dict[str, Any]]) -> str:
        t = transitions[0]
        return f"When '{t.get('failed_action')}' fails, use '{t.get('fix_action')}' instead."


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI, Groq, Together, DeepSeek, or any /v1/chat/completions endpoint."""

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("SMRUTI_LLM_MODEL") or os.getenv("smruti_LLM_MODEL") or "gpt-4o-mini"
        self.base_url = (
            base_url
            or os.getenv("SMRUTI_LLM_BASE_URL")
            or os.getenv("smruti_LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _call(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        data = _post_json(url, headers, payload)
        return data["choices"][0]["message"]["content"]

    def evaluate_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        prompt = (
            f"You are a preflight cognitive safety gate for an autonomous AI agent.\n"
            f"Candidate Action: '{action}'\n"
            f"Known Failure Signature: '{anti_memory.signature}'\n"
            f"Pattern: '{anti_memory.pattern}'\n"
            f"Prior Failure Reason: '{anti_memory.reason}'\n\n"
            f"Determine if this candidate action will trigger the known failure. "
            f"Reply strictly with JSON: {{\"block\": true/false, \"explanation\": \"short reason\"}}"
        )
        try:
            res_text = self._call(prompt)
            data = json.loads(res_text)
            return (bool(data.get("block", True)), data.get("explanation"))
        except Exception as e:
            logger.warning("OpenAI preflight evaluation failed: %s; falling back to block", e)
            return (True, anti_memory.reason)

    def summarize_failure(self, cluster_data: list[dict[str, Any]]) -> str:
        prompt = (
            f"Summarize these repeated tool/command failures into a crisp, single-sentence failure invariant:\n"
            f"{json.dumps(cluster_data, indent=2)}\n\n"
            f"Output only the distilled reason."
        )
        try:
            return self._call(prompt).strip()
        except Exception as e:
            logger.warning("OpenAI failure summarization failed: %s", e)
            return HeuristicFallbackProvider().summarize_failure(cluster_data)

    def distill_resolution(self, transitions: list[dict[str, Any]]) -> str:
        prompt = (
            f"An agent failed at an action, then successfully solved it with an alternative action.\n"
            f"Transitions: {json.dumps(transitions, indent=2)}\n\n"
            f"Synthesize this into a reusable heuristic rule (e.g., 'When X fails because of Y, do Z')."
        )
        try:
            return self._call(prompt).strip()
        except Exception as e:
            logger.warning("OpenAI resolution distillation failed: %s", e)
            return HeuristicFallbackProvider().distill_resolution(transitions)


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models via /v1/messages."""

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("SMRUTI_LLM_MODEL") or os.getenv("smruti_LLM_MODEL") or "claude-3-5-haiku-latest"
        self.base_url = (base_url or os.getenv("SMRUTI_LLM_BASE_URL") or "https://api.anthropic.com/v1").rstrip("/")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _call(self, prompt: str) -> str:
        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self.model,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        data = _post_json(url, headers, payload)
        return data["content"][0]["text"]

    def evaluate_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        prompt = (
            f"You are a preflight cognitive safety gate for an autonomous AI agent.\n"
            f"Candidate Action: '{action}'\n"
            f"Known Failure Signature: '{anti_memory.signature}'\n"
            f"Pattern: '{anti_memory.pattern}'\n"
            f"Prior Failure Reason: '{anti_memory.reason}'\n\n"
            f"Determine if this candidate action will trigger the known failure. "
            f"Reply strictly with JSON: {{\"block\": true/false, \"explanation\": \"short reason\"}}"
        )
        try:
            res_text = self._call(prompt)
            data = json.loads(res_text)
            return (bool(data.get("block", True)), data.get("explanation"))
        except Exception as e:
            logger.warning("Anthropic preflight evaluation failed: %s; falling back to block", e)
            return (True, anti_memory.reason)

    def summarize_failure(self, cluster_data: list[dict[str, Any]]) -> str:
        prompt = (
            f"Summarize these repeated tool/command failures into a crisp, single-sentence failure invariant:\n"
            f"{json.dumps(cluster_data, indent=2)}\n\n"
            f"Output only the distilled reason."
        )
        try:
            return self._call(prompt).strip()
        except Exception as e:
            logger.warning("Anthropic failure summarization failed: %s", e)
            return HeuristicFallbackProvider().summarize_failure(cluster_data)

    def distill_resolution(self, transitions: list[dict[str, Any]]) -> str:
        prompt = (
            f"An agent failed at an action, then successfully solved it with an alternative action.\n"
            f"Transitions: {json.dumps(transitions, indent=2)}\n\n"
            f"Synthesize this into a reusable heuristic rule (e.g., 'When X fails because of Y, do Z')."
        )
        try:
            return self._call(prompt).strip()
        except Exception as e:
            logger.warning("Anthropic resolution distillation failed: %s", e)
            return HeuristicFallbackProvider().distill_resolution(transitions)


class OllamaProvider(BaseLLMProvider):
    """Local Ollama instance via /api/chat."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("SMRUTI_LLM_MODEL") or "llama3.2:latest"

    def is_available(self) -> bool:
        # Check if Ollama endpoint is configured or reachable
        return bool(os.getenv("OLLAMA_HOST"))

    def _call(self, prompt: str) -> str:
        url = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        data = _post_json(url, headers, payload)
        return data["message"]["content"]

    def evaluate_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        try:
            res_text = self._call(
                f"Candidate Action: '{action}'\nFailure: '{anti_memory.pattern}'. Should this action be blocked? "
                f"Respond with JSON: {{\"block\": true/false, \"explanation\": \"reason\"}}"
            )
            data = json.loads(res_text)
            return (bool(data.get("block", True)), data.get("explanation"))
        except Exception:
            return (True, anti_memory.reason)

    def summarize_failure(self, cluster_data: list[dict[str, Any]]) -> str:
        try:
            return self._call(f"Summarize these repeated tool failures into one crisp rule: {json.dumps(cluster_data)}").strip()
        except Exception:
            return HeuristicFallbackProvider().summarize_failure(cluster_data)

    def distill_resolution(self, transitions: list[dict[str, Any]]) -> str:
        try:
            return self._call(f"Distill this failure-fix transition into a heuristic rule: {json.dumps(transitions)}").strip()
        except Exception:
            return HeuristicFallbackProvider().distill_resolution(transitions)


class LLMClient:
    """
    High-level facade that selects the active LLM provider based on environment configuration.
    Priority:
    1. Explicit custom provider (if injected)
    2. Anthropic (if ANTHROPIC_API_KEY is present)
    3. Ollama (if OLLAMA_HOST is present)
    4. OpenAI-Compatible (if OPENAI_API_KEY or SMRUTI_LLM_BASE_URL is present)
    5. HeuristicFallbackProvider (zero-setup offline default)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider: BaseLLMProvider | None = None
    ):
        if provider is not None:
            self.provider = provider
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.provider = AnthropicProvider(api_key=api_key, model=model, base_url=base_url)
        elif os.getenv("OLLAMA_HOST"):
            self.provider = OllamaProvider(host=base_url, model=model)
        elif api_key or os.getenv("OPENAI_API_KEY") or base_url or os.getenv("SMRUTI_LLM_BASE_URL"):
            self.provider = OpenAICompatibleProvider(api_key=api_key, model=model, base_url=base_url)
        else:
            self.provider = HeuristicFallbackProvider()

    def is_configured(self) -> bool:
        """Returns True if a real neural LLM provider is active (not heuristic fallback)."""
        return not isinstance(self.provider, HeuristicFallbackProvider) and self.provider.is_available()

    def get_provider_name(self) -> str:
        return self.provider.__class__.__name__

    def evaluate_action_preflight(self, action: str, anti_memory: AntiMemory) -> tuple[bool, str | None]:
        return self.provider.evaluate_preflight(action, anti_memory)

    def summarize_failure_cluster(self, cluster_data: list[dict[str, Any]]) -> str:
        return self.provider.summarize_failure(cluster_data)

    def distill_resolution_heuristic(self, transitions: list[dict[str, Any]]) -> str:
        return self.provider.distill_resolution(transitions)
