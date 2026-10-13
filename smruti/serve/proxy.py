"""
proxy.py - OpenAI-compatible memory proxy. Sits between any chat client and any
OpenAI-compatible runtime: recalled memory is merged into each chat request, new user and
tool turns plus the reply are recorded, and every other /v1 endpoint passes through untouched.
Memory failures never break a request: recall errors forward the original request, recording
errors are logged.
"""

import json
import logging
from contextlib import asynccontextmanager

import anyio
import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from smruti.memory import Memory
from smruti.serve.capture import StreamAccumulator, cap_text, mask_secrets, new_turns, reply_text
from smruti.serve.scheduler import SleepScheduler

logger = logging.getLogger(__name__)

SPACE_HEADER = "x-smruti-space"
SKIP_HEADER = "x-smruti-skip"
_DROPPED_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding",
                    SPACE_HEADER, SKIP_HEADER}
_HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "te",
               "trailer", "upgrade"}


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items()
            if k.lower() not in _HOP_BY_HOP and not k.lower().startswith("proxy-")}


def create_app(memory: Memory, upstream: str, default_space: str = "default",
               sleep_poll_seconds: float | None = 30.0, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=httpx.Timeout(None, connect=10.0),
                               transport=transport)
    scheduler = SleepScheduler(memory, sleep_poll_seconds) if sleep_poll_seconds else None

    def forwarded_headers(request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in _DROPPED_HEADERS}

    def record(space: str, turns: list[tuple[str, str]], reply: str) -> None:
        try:
            for role, text in turns:
                if role == "tool":
                    text = cap_text(text, memory.config.max_tool_chars)
                memory.observe(role, mask_secrets(text), space)
            if reply:
                memory.observe("assistant", mask_secrets(reply), space)
        except Exception:
            logger.exception("recording turns failed for space %r", space)

    def upstream_error(exc: Exception) -> JSONResponse:
        return JSONResponse({"error": {"message": f"upstream unreachable: {exc}", "type": "upstream_error"}},
                            status_code=502)

    async def relay_response(response: httpx.Response) -> Response:
        """Relays an upstream response (opened with stream=True): event streams as they arrive, the rest buffered."""
        headers = _response_headers(response)
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            async def relay():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                finally:
                    with anyio.CancelScope(shield=True):
                        await response.aclose()

            return StreamingResponse(relay(), status_code=response.status_code, headers=headers)
        try:
            content = await response.aread()
        finally:
            await response.aclose()
        return Response(content, status_code=response.status_code, headers=headers)

    async def send_upstream(method: str, path: str, **kwargs) -> httpx.Response | JSONResponse:
        try:
            return await client.send(client.build_request(method, path, **kwargs), stream=True)
        except httpx.HTTPError as exc:
            return upstream_error(exc)

    async def forward_chat_unchanged(raw: bytes, request: Request) -> Response:
        response = await send_upstream("POST", "/chat/completions", content=raw, headers=forwarded_headers(request))
        return response if isinstance(response, JSONResponse) else await relay_response(response)

    def skip_requested(request: Request) -> bool:
        return request.headers.get(SKIP_HEADER, "").strip() not in ("", "0")

    async def chat_completions(request: Request) -> Response:
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            return JSONResponse({"error": {"message": "Content-Type must be application/json",
                                           "type": "invalid_request_error"}}, status_code=415)
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or skip_requested(request):
            return await forward_chat_unchanged(raw, request)  # not ours to interpret: fail open, no memory

        messages = body["messages"]
        space = request.headers.get(SPACE_HEADER) or body.get("user") or default_space
        if memory.model is None and body.get("model"):
            memory.model = body["model"]  # sleep consolidates with the model clients use
            logger.info("sleep will use model %s; pass --model to choose", memory.model)
        try:
            body = {**body, "messages": await run_in_threadpool(memory.build_context, messages, space)}
        except Exception:
            logger.exception("recall failed; forwarding the request unchanged")
        try:
            turns = new_turns(messages)
        except Exception:
            logger.exception("reading new turns failed; nothing will be recorded")
            turns = []

        response = await send_upstream("POST", "/chat/completions", json=body, headers=forwarded_headers(request))
        if isinstance(response, JSONResponse):
            return response

        if response.status_code >= 400 or not body.get("stream"):
            content = await response.aread()
            await response.aclose()
            task = BackgroundTask(record, space, turns, reply_text(content)) if response.status_code < 400 else None
            return Response(content, status_code=response.status_code, headers=_response_headers(response),
                            background=task)

        await run_in_threadpool(record, space, turns, "")  # upstream accepted it: the turns count even if the client leaves
        accumulator = StreamAccumulator()

        async def relay():
            try:
                async for chunk in response.aiter_bytes():
                    accumulator.feed(chunk)
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):  # runs on client disconnect too, with whatever was streamed
                    await response.aclose()
                    await run_in_threadpool(record, space, [], accumulator.text)

        return StreamingResponse(relay(), status_code=response.status_code, headers=_response_headers(response))

    async def passthrough(request: Request) -> Response:
        response = await send_upstream(request.method, "/" + request.path_params["path"],
                                       content=await request.body(), headers=forwarded_headers(request),
                                       params=request.query_params)
        return response if isinstance(response, JSONResponse) else await relay_response(response)

    @asynccontextmanager
    async def lifespan(app):
        if scheduler is not None:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()
            await client.aclose()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/{path:path}", passthrough, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
        ],
        lifespan=lifespan,
    )
