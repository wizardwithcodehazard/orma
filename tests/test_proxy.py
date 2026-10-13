import json

import httpx
import pytest
from starlette.testclient import TestClient

from smruti import Memory
from smruti.serve.proxy import create_app


class Upstream:
    """Fake OpenAI-compatible server that records what it received."""

    def __init__(self, reply="Hello!", status=200, stream_chunks=None):
        self.requests = []
        self.reply = reply
        self.status = status
        self.stream_chunks = stream_chunks or []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append({"method": request.method, "path": request.url.path, "body": body,
                              "headers": dict(request.headers)})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "llama3.2:3b"}]})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "bad"}})
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=b"".join(self.stream_chunks))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": self.reply}}]})


@pytest.fixture
def memory(config, embedder):
    mem = Memory(config=config, embedder=embedder)
    yield mem
    mem.close()


def client(memory, upstream):
    app = create_app(memory, "http://up/v1", sleep_poll_seconds=None, transport=httpx.MockTransport(upstream))
    return TestClient(app)


def chat(c, content="hi", **extra):
    return c.post("/v1/chat/completions",
                  json={"model": "llama3.2:3b", "messages": [{"role": "user", "content": content}], **extra})


def texts(memory, space="default"):
    return [(e.role, e.text) for e in memory.store.pending_episodes(space, 20)]


def test_injects_memory_and_records_the_exchange(memory):
    s = memory.store
    user = s.user_entity("default")
    s.add_fact("default", user, s.add_predicate("default", "lives_in", "one"),
               s.add_entity("default", "Pune", "city"), None, 1.0, [1])
    up = Upstream(reply="Try Lonavala.")
    with client(memory, up) as c:
        r = chat(c, "Weekend trip ideas?")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Try Lonavala."
    sent = up.requests[0]["body"]["messages"]
    assert sent[0]["role"] == "system" and "user lives in Pune" in sent[0]["content"]
    assert up.requests[0]["path"] == "/v1/chat/completions"
    assert texts(memory) == [("user", "Weekend trip ideas?"), ("assistant", "Try Lonavala.")]
    assert memory.model == "llama3.2:3b"


def test_space_comes_from_header_then_user_field(memory):
    up = Upstream()
    with client(memory, up) as c:
        c.post("/v1/chat/completions", headers={"X-Smruti-Space": "work"},
               json={"model": "m", "messages": [{"role": "user", "content": "a"}]})
        c.post("/v1/chat/completions", json={"model": "m", "user": "alice",
                                             "messages": [{"role": "user", "content": "b"}]})
    assert texts(memory, "work") == [("user", "a"), ("assistant", "Hello!")]
    assert texts(memory, "alice") == [("user", "b"), ("assistant", "Hello!")]
    assert "x-smruti-space" not in up.requests[0]["headers"]


def test_tool_results_are_recorded_once(memory):
    messages = [
        {"role": "user", "content": "build it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_shell", "arguments": "npm run build"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Error: Cannot find module 'vite'"},
    ]
    with client(memory, Upstream(reply="Install vite first.")) as c:
        c.post("/v1/chat/completions", json={"model": "m", "messages": messages})
    assert texts(memory) == [("tool", "run_shell(npm run build) -> Error: Cannot find module 'vite'"),
                             ("assistant", "Install vite first.")]


def test_streaming_is_relayed_byte_for_byte_and_recorded(memory):
    chunks = [b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
              b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
              b"data: [DONE]\n\n"]
    with client(memory, Upstream(stream_chunks=chunks)) as c:
        r = chat(c, "hi", stream=True)
        assert r.status_code == 200
        assert r.content == b"".join(chunks)
    assert texts(memory) == [("user", "hi"), ("assistant", "Hello")]


def test_upstream_errors_pass_through_and_nothing_is_recorded(memory):
    with client(memory, Upstream(status=400)) as c:
        r = chat(c)
    assert r.status_code == 400 and r.json()["error"]["message"] == "bad"
    assert texts(memory) == []


def test_unreachable_upstream_returns_502(memory):
    def down(request):
        raise httpx.ConnectError("refused")

    with client(memory, down) as c:
        r = chat(c)
    assert r.status_code == 502 and "upstream" in r.json()["error"]["message"]


def test_memory_failure_forwards_the_original_request(memory, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(memory, "build_context", boom)
    up = Upstream()
    with client(memory, up) as c:
        r = chat(c, "hi")
    assert r.status_code == 200
    assert up.requests[0]["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_other_endpoints_pass_through(memory):
    up = Upstream()
    with client(memory, up) as c:
        r = c.get("/v1/models")
    assert r.json()["data"][0]["id"] == "llama3.2:3b"
    assert up.requests[0]["path"] == "/v1/models"


def raw_post(c, content, content_type="application/json", **headers):
    return c.post("/v1/chat/completions", content=content, headers={"Content-Type": content_type, **headers})


def test_invalid_json_is_forwarded_unchanged(memory):
    seen = []

    def up(request):
        seen.append(request.content)
        return httpx.Response(418, content=b"teapot")

    with client(memory, up) as c:
        r = raw_post(c, b"{not json")
    assert r.status_code == 418 and r.content == b"teapot"
    assert seen == [b"{not json"]
    assert texts(memory) == [] and memory.model is None


def test_non_list_messages_are_forwarded_unchanged(memory):
    seen = []

    def up(request):
        seen.append(request.content)
        return httpx.Response(400, json={"error": {"message": "bad"}})

    raw = b'{"model": "m", "messages": "hello"}'
    with client(memory, up) as c:
        r = raw_post(c, raw)
        r2 = raw_post(c, b"[1, 2]")
    assert r.status_code == 400 and r2.status_code == 400
    assert seen == [raw, b"[1, 2]"]
    assert texts(memory) == []


def test_new_turns_failure_does_not_break_the_request(memory):
    up = Upstream()
    with client(memory, up) as c:
        r = c.post("/v1/chat/completions", json={"model": "m", "messages": ["oops", {"role": "user", "content": "x"}]})
    assert r.status_code == 200


def test_skip_header_means_no_recall_no_record_no_model(memory):
    up = Upstream()
    with client(memory, up) as c:
        r = c.post("/v1/chat/completions", headers={"x-smruti-skip": "1"},
                   json={"model": "title-model", "messages": [{"role": "user", "content": "Title?"}]})
        assert r.status_code == 200
        assert "x-smruti-skip" not in up.requests[0]["headers"]
        assert up.requests[0]["body"]["messages"] == [{"role": "user", "content": "Title?"}]
        c.post("/v1/chat/completions", headers={"X-Smruti-Skip": "0"},
               json={"model": "real", "messages": [{"role": "user", "content": "kept"}]})
    assert texts(memory) == [("user", "kept"), ("assistant", "Hello!")]
    assert memory.model == "real"


def test_streamed_passthrough_is_relayed_byte_for_byte(memory):
    chunks = [b'data: {"choices":[{"text":"a"}]}\n\n', b"data: [DONE]\n\n"]

    def up(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream", "x-upstream": "1"},
                              content=b"".join(chunks))

    with client(memory, up) as c:
        r = c.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": True})
    assert r.content == b"".join(chunks)
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["x-upstream"] == "1"


def test_upstream_headers_are_copied_except_hop_by_hop(memory):
    def up(request):
        return httpx.Response(200, json={"choices": []}, headers={"x-ratelimit-remaining": "9", "keep-alive": "x"})

    with client(memory, up) as c:
        r = chat(c)
    assert r.headers["x-ratelimit-remaining"] == "9" and "keep-alive" not in r.headers


def test_non_json_content_type_is_rejected(memory):
    up = Upstream()
    body = b'{"model": "m", "messages": [{"role": "user", "content": "x"}]}'
    with client(memory, up) as c:
        for ctype in ("text/plain", "application/x-www-form-urlencoded", ""):
            r = raw_post(c, body, ctype)
            assert r.status_code == 415 and r.json()["error"]["type"] == "invalid_request_error"
        ok = raw_post(c, body, "application/json; charset=utf-8")
    assert ok.status_code == 200 and len(up.requests) == 1


def test_secrets_are_masked_and_tool_output_is_capped(memory):
    memory.config.max_tool_chars = 100
    messages = [
        {"role": "user", "content": "my key is sk-abcdefghijklmnopqrstu"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "sh", "arguments": "cat big"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "HEAD" + "x" * 5000 + "TAIL"},
    ]
    with client(memory, Upstream(reply="ok")) as c:
        c.post("/v1/chat/completions", json={"model": "m", "messages": messages[:1]})
        c.post("/v1/chat/completions", json={"model": "m", "messages": messages})
    recorded = texts(memory)
    assert recorded[0] == ("user", "my key is [REDACTED]")
    tool = next(t for r, t in recorded if r == "tool")
    assert "…[truncated]…" in tool and tool.endswith("TAIL") and len(tool) < 200


def test_stream_records_the_user_turn_and_partial_reply_on_client_disconnect(memory):
    import asyncio

    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            await asyncio.sleep(30)
            yield b"data: [DONE]\n\n"

    def up(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Slow())

    app = create_app(memory, "http://up/v1", sleep_poll_seconds=None, transport=httpx.MockTransport(up))
    body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()

    async def run():
        first_chunk = asyncio.Event()
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            await first_chunk.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_chunk.set()

        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
                 "query_string": b"", "headers": [(b"content-type", b"application/json"),
                                                  (b"content-length", str(len(body)).encode())],
                 "http_version": "1.1", "scheme": "http", "server": ("t", 80), "client": ("c", 1), "root_path": ""}
        await asyncio.wait_for(app(scope, receive, send), 10)

    asyncio.run(run())
    assert texts(memory) == [("user", "hi"), ("assistant", "Hel")]
