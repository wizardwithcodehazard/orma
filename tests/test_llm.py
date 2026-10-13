import json

import httpx
import pytest

from smruti.llm import LLMClient, LLMError


def client(handler):
    return LLMClient("http://local/v1", model="m", transport=httpx.MockTransport(handler))


def reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_chat_posts_openai_payload():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return reply("hi")

    assert client(handler).chat([{"role": "user", "content": "yo"}]) == "hi"
    assert seen["url"] == "http://local/v1/chat/completions"
    assert seen["body"]["model"] == "m" and seen["body"]["stream"] is False
    assert "response_format" not in seen["body"]


def test_chat_json_retries_once_with_stricter_reminder_at_temperature_zero():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return reply("not json" if len(bodies) == 1 else '{"facts": []}')

    assert client(handler).chat_json([]) == {"facts": []}
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[0]["temperature"] == 0.0 and bodies[1]["temperature"] == 0.0
    assert bodies[1]["messages"][-1] == {"role": "user", "content": "Return only a valid JSON object."}
    assert len(bodies[1]["messages"]) == len(bodies[0]["messages"]) + 1


def test_chat_json_extracts_object_from_fenced_output():
    fenced = 'Sure!\n```json\n{"facts": []}\n```'
    assert client(lambda request: reply(fenced)).chat_json([]) == {"facts": []}


def test_chat_json_gives_up_after_retry():
    with pytest.raises(LLMError):
        client(lambda request: reply("nope")).chat_json([])


def test_http_errors_become_llm_errors():
    with pytest.raises(LLMError):
        client(lambda request: httpx.Response(500, json={"error": "boom"})).chat([])


def test_http_error_includes_body_snippet():
    with pytest.raises(LLMError, match="model not loaded"):
        client(lambda request: httpx.Response(500, text="model not loaded")).chat([])


def test_non_string_content_raises():
    with pytest.raises(LLMError):
        client(lambda request: reply([{"type": "text"}])).chat([])
