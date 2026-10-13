"""
proxy.py - OpenAI-compatible memory proxy. Sits between any chat client and any
OpenAI-compatible runtime: recalled memory is merged into each chat request, new user and
tool turns plus the reply are recorded, and every other /v1 endpoint passes through untouched.
Memory failures never break a request: recall errors forward the original request, recording
errors are logged.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from smruti.memory import Memory
from smruti.serve.capture import StreamAccumulator, new_turns, reply_text
from smruti.serve.scheduler import SleepScheduler

logger = logging.getLogger(__name__)

SPACE_HEADER = "x-smruti-space"
_DROPPED_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding", SPACE_HEADER}


def create_app(memory: Memory, upstream: str, default_space: str = "default",
               sleep_poll_seconds: float | None = 30.0, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=None, transport=transport)
    scheduler = SleepScheduler(memory, sleep_poll_seconds) if sleep_poll_seconds else None

    def forwarded_headers(request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in _DROPPED_HEADERS}

    def record(space: str, turns: list[tuple[str, str]], reply: str) -> None:
        try:
            for role, text in turns:
                memory.observe(role, text, space)
            if reply:
                memory.observe("assistant", reply, space)
        except Exception:
            logger.exception("recording turns failed for space %r", space)

    def upstream_error(exc: Exception) -> JSONResponse:
        return JSONResponse({"error": {"message": f"upstream unreachable: {exc}", "type": "upstream_error"}},
                            status_code=502)

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        messages = body.get("messages") or []
        space = request.headers.get(SPACE_HEADER) or body.get("user") or default_space
        if memory.model is None and body.get("model"):
            memory.model = body["model"]  # sleep consolidates with the model clients use
        try:
            body = {**body, "messages": await run_in_threadpool(memory.build_context, messages, space)}
        except Exception:
            logger.exception("recall failed; forwarding the request unchanged")
        turns = new_turns(messages)

        outgoing = client.build_request("POST", "/chat/completions", json=body, headers=forwarded_headers(request))
        try:
            response = await client.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            return upstream_error(exc)

        content_type = response.headers.get("content-type")
        if response.status_code >= 400 or not body.get("stream"):
            content = await response.aread()
            await response.aclose()
            task = BackgroundTask(record, space, turns, reply_text(content)) if response.status_code < 400 else None
            return Response(content, status_code=response.status_code, media_type=content_type, background=task)

        accumulator = StreamAccumulator()

        async def relay():
            try:
                async for chunk in response.aiter_bytes():
                    accumulator.feed(chunk)
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(relay(), status_code=response.status_code,
                                 media_type=content_type or "text/event-stream",
                                 background=BackgroundTask(lambda: record(space, turns, accumulator.text)))

    async def passthrough(request: Request) -> Response:
        outgoing = client.build_request(request.method, "/" + request.path_params["path"],
                                        content=await request.body(), headers=forwarded_headers(request),
                                        params=request.query_params)
        try:
            response = await client.send(outgoing)
        except httpx.HTTPError as exc:
            return upstream_error(exc)
        return Response(response.content, status_code=response.status_code,
                        media_type=response.headers.get("content-type"))

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
