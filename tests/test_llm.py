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


SCHEMA = {"type": "object", "properties": {"facts": {"type": "array"}}, "required": ["facts"]}


def test_chat_json_with_schema_sends_strict_json_schema_and_given_temperature():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return reply("not json" if len(bodies) == 1 else '{"facts": []}')

    assert client(handler).chat_json([], schema=SCHEMA, temperature=0.7) == {"facts": []}
    expected = {"type": "json_schema", "json_schema": {"name": "memory", "schema": SCHEMA, "strict": True}}
    assert bodies[0]["response_format"] == expected and bodies[1]["response_format"] == expected
    assert bodies[0]["temperature"] == 0.7 and bodies[1]["temperature"] == 0.0


def test_json_schema_rejected_with_4xx_falls_back_to_json_object_and_is_remembered():
    formats = []

    def handler(request):
        body = json.loads(request.content)
        formats.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="response_format json_schema not supported")
        return reply('{"facts": []}')

    llm = client(handler)
    assert llm.chat_json([], schema=SCHEMA) == {"facts": []}
    assert llm.chat_json([], schema=SCHEMA) == {"facts": []}
    assert formats == ["json_schema", "json_object", "json_object"]


def test_json_schema_server_error_is_not_a_fallback():
    with pytest.raises(LLMError):
        client(lambda request: httpx.Response(500, text="boom")).chat_json([], schema=SCHEMA)
