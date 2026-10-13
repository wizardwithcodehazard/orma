"""
capture.py - What to record from proxied traffic. Clients resend the whole conversation on
every request, so only the turns after the last assistant message are new: the user's latest
message and any tool results (paired with the call that produced them). Replies are read from
JSON responses or accumulated from SSE streams as they pass through.
"""

import codecs
import json
import re

from smruti.messages import message_text

_REDACTED = "[REDACTED]"
_SECRET_PATTERNS = [  # matched by format only
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), _REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), _REDACTED),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), f"Bearer {_REDACTED}"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), _REDACTED),
    (re.compile(r"((?:password|passwd|secret|token|api[_-]?key)\s*=\s*)[^\s&\"']+", re.IGNORECASE),
     rf"\1{_REDACTED}"),
]
_TRUNCATED = "…[truncated]…"


def mask_secrets(text: str) -> str:
    """Replaces credentials (recognised by their format) with [REDACTED] before text is stored."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def cap_text(text: str, limit: int) -> str:
    """Keeps the head and tail of text longer than limit, with a marker in between."""
    if len(text) <= limit:
        return text
    head = limit // 2
    return text[:head] + _TRUNCATED + text[len(text) - (limit - head):]


def _call_label(call: dict) -> str:
    function = call.get("function") or {}
    return f"{function.get('name', 'tool')}({function.get('arguments', '')})"


def new_turns(messages: list[dict]) -> list[tuple[str, str]]:
    """(role, text) for the new user and tool turns of a request."""
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    calls = {}
    if last_assistant >= 0:
        calls = {c.get("id"): _call_label(c) for c in messages[last_assistant].get("tool_calls") or []}
    turns = []
    for message in messages[last_assistant + 1:]:
        role, text = message.get("role"), message_text(message).strip()
        if role == "user" and text:
            turns.append(("user", text))
        elif role == "tool":
            turns.append(("tool", f"{calls.get(message.get('tool_call_id'), 'tool call')} -> {text}"))
    return turns


def reply_text(body: bytes) -> str:
    """Assistant text from a non-streaming chat completion body ('' if absent or unparsable)."""
    try:
        return message_text(json.loads(body)["choices"][0]["message"]).strip()
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


class StreamAccumulator:
    """Collects assistant text from OpenAI-style server-sent event chunks."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._parts: list[str] = []

    def feed(self, chunk: bytes) -> None:
        self._pending += self._decoder.decode(chunk)
        *lines, self._pending = self._pending.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                choices = json.loads(line[5:])["choices"]
                deltas = [c.get("delta") or {} for c in choices if c.get("index", 0) == 0]
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            for delta in deltas:
                if isinstance(delta.get("content"), str):
                    self._parts.append(delta["content"])

    @property
    def text(self) -> str:
        return "".join(self._parts).strip()
