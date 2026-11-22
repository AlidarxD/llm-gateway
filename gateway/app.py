"""FastAPI application: /v1/chat/completions, /v1/models, health and metrics."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.auth import KeyInfo, KeyStore
from gateway.config import Settings
from gateway.db import UsageLog, make_engine, make_sessionmaker
from gateway.errors import AllProvidersFailed, ProviderError, UnknownModel
from gateway.limits import Limiter
from gateway.logging import request_id_var, setup_logging
from gateway.metrics import Metrics
from gateway.middleware.base import Middleware
from gateway.pipeline import Pipeline
from gateway.providers import build_providers
from gateway.providers.base import Provider
from gateway.router import RouteInfo, Router
from gateway.schemas import (
    BudgetInfo,
    ChatChoice,
    ChatChunk,
    ChatChunkChoice,
    ChatDelta,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    RequestContext,
    Usage,
)
from gateway.usage import UsageWriter
from gateway.wiring import default_middlewares

log = logging.getLogger("gateway.app")


def _err(
    status: int, message: str, code: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": code, "code": code}},
        status_code=status,
        headers=headers,
    )


def _sse_line(c: ChatChunk) -> bytes:
    return f"data: {c.model_dump_json(exclude_none=True)}\n\n".encode()


def response_to_chunks(resp: ChatResponse) -> list[ChatChunk]:
    """Replay a full response as stream chunks (used for cache hits on stream=true)."""
    base = {"id": resp.id, "created": resp.created, "model": resp.model}
    out = [
        ChatChunk(**base, choices=[ChatChunkChoice(delta=ChatDelta(role="assistant", content=""))])
    ]
    for c in resp.choices:
        text = c.message.content if isinstance(c.message.content, str) else ""
        out.append(
            ChatChunk(
                **base, choices=[ChatChunkChoice(index=c.index, delta=ChatDelta(content=text))]
            )
        )
        out.append(
            ChatChunk(
                **base,
                choices=[
                    ChatChunkChoice(index=c.index, delta=ChatDelta(), finish_reason=c.finish_reason)
                ],
                usage=resp.usage,
            )
        )
    return out


def aggregate_chunks(chunks: list[ChatChunk], model: str, usage: Usage) -> ChatResponse:
    text = "".join(ch.delta.content or "" for c in chunks for ch in c.choices)
    finish = next(
        (ch.finish_reason for c in reversed(chunks) for ch in c.choices if ch.finish_reason), "stop"
    )
    first = chunks[0] if chunks else None
    return ChatResponse(
        id=first.id if first else f"chatcmpl-{uuid.uuid4().hex}",
        created=first.created if first else int(time.time()),
        model=model,
        choices=[
            ChatChoice(message=ChatMessage(role="assistant", content=text), finish_reason=finish)
        ],
        usage=usage,
    )


def create_app(
    settings: Settings | None = None,
    *,
    redis: Redis | None = None,
    session_maker: async_sessionmaker[AsyncSession] | None = None,
    providers: dict[str, Provider] | None = None,
    middlewares: list[Middleware] | None = None,
    metrics: Metrics | None = None,
    sleep: Callable[[float], object] | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        http = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_s))
        engine = None
        sm = session_maker
        if sm is None:
            engine = make_engine(settings.database_url)
            sm = make_sessionmaker(engine)
        r = redis or Redis.from_url(settings.redis_url)
        m = metrics or Metrics()
        provs = providers if providers is not None else build_providers(settings, http)
        kwargs = {"sleep": sleep} if sleep else {}
        app.state.router = Router(provs, settings.load_routes(), settings, m, **kwargs)  # type: ignore[arg-type]
        app.state.pipeline = Pipeline(
            middlewares if middlewares is not None else default_middlewares(settings, r, m)
        )
        app.state.keys = KeyStore(sm, settings.auth_cache_ttl_s)
        app.state.limiter = Limiter(r)
        app.state.metrics = m
        app.state.sm = sm
        app.state.usage = UsageWriter(sm)
        app.state.usage.start()
        app.state.redis = r
        app.state.tasks = set()
        yield
        if app.state.tasks:
            await asyncio.gather(*app.state.tasks, return_exceptions=True)
        await app.state.usage.close()
        await http.aclose()
        if redis is None:
            await r.aclose()
        if engine is not None:
            await engine.dispose()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            resp = await call_next(request)
        finally:
            request_id_var.reset(token)
        resp.headers["x-request-id"] = rid
        return resp

    @app.exception_handler(UnknownModel)
    async def _unknown(_: Request, e: UnknownModel) -> JSONResponse:
        return _err(404, str(e), e.code)

    @app.exception_handler(AllProvidersFailed)
    async def _all_failed(_: Request, e: AllProvidersFailed) -> JSONResponse:
        return _err(502, str(e), e.code)

    @app.exception_handler(ProviderError)
    async def _perr(_: Request, e: ProviderError) -> JSONResponse:
        return _err(e.status_code, e.message or str(e), e.code)

    # ---- helpers ---------------------------------------------------------
    async def authenticate(request: Request) -> KeyInfo | JSONResponse:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return _err(401, "Missing bearer token", "invalid_api_key")
        info: KeyInfo | None = await request.app.state.keys.lookup(auth[7:].strip())
        if info is None:
            return _err(401, "Invalid API key", "invalid_api_key")
        return info

    def spawn(coro) -> None:  # type: ignore[no-untyped-def]
        t = asyncio.create_task(coro)
        app.state.tasks.add(t)
        t.add_done_callback(app.state.tasks.discard)

    async def record(
        *,
        key: KeyInfo,
        ctx: RequestContext,
        requested_model: str,
        info: RouteInfo,
        status: int,
        started: float,
        cache_hit: bool,
        stream: bool,
        error: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        usage = usage or info.usage
        latency = time.monotonic() - started
        m: Metrics = app.state.metrics
        m.requests.labels(str(status), requested_model).inc()
        m.request_latency.observe(latency)
        cost = Decimal(0) if cache_hit else info.cost_usd
        try:
            if not cache_hit and usage.total_tokens:
                await app.state.limiter.record(key, usage.total_tokens, cost)
            app.state.usage.add(
                UsageLog(
                    request_id=ctx.request_id,
                    api_key_id=key.id,
                    tenant_id=key.tenant_id,
                    requested_model=requested_model,
                    provider=info.provider or None,
                    upstream_model=info.upstream_model or None,
                    status=status,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cost_usd=cost,
                    latency_ms=int(latency * 1000),
                    cache_hit=cache_hit,
                    fallbacks=info.fallbacks,
                    retries=info.retries,
                    stream=stream,
                    error=error,
                )
            )
        except Exception:
            log.exception("failed to record usage")
        log.info(
            "request",
            extra={
                "tenant_id": key.tenant_id,
                "api_key_id": key.id,
                "model": requested_model,
                "provider": info.provider,
                "status": status,
                "latency_ms": int(latency * 1000),
                "tokens": usage.total_tokens,
                "cost_usd": str(cost),
                "fallbacks": info.fallbacks,
                "cache_hit": cache_hit,
            },
        )

    def note_cache(ctx: RequestContext) -> bool:
        hit = ctx.extra.get("cache_hit")
        if hit is not None:
            app.state.metrics.cache.labels("hit" if hit else "miss").inc()
        return bool(hit)

    # ---- routes ----------------------------------------------------------
    @app.post("/v1/chat/completions", response_model=None)
    async def chat(req: ChatRequest, request: Request) -> Response:
        started = time.monotonic()
        key = await authenticate(request)
        if isinstance(key, JSONResponse):
            return key
        decision = await app.state.limiter.check(key)
        if not decision.allowed:
            app.state.metrics.rate_limited.labels(decision.reason).inc()
            status = 429
            return _err(
                status,
                f"Limit exceeded: {decision.reason}",
                decision.reason or "rate_limit",
                {"Retry-After": str(decision.retry_after_s)},
            )
        ctx = RequestContext(
            api_key_id=key.id,
            tenant_id=key.tenant_id,
            request_id=request.state.request_id,
            budget=decision.budget or BudgetInfo(),
        )
        info = RouteInfo()
        pipeline: Pipeline = app.state.pipeline
        router: Router = app.state.router
        model = req.model

        if not req.stream:
            try:
                req2, short, ran = await pipeline.run_before(req, ctx)
                cache_hit = short is not None
                resp = short if short is not None else await router.complete(req2, info)
                resp = await pipeline.run_after(req2, resp, ctx, ran)
            except (UnknownModel, ProviderError, AllProvidersFailed) as e:
                await record(
                    key=key,
                    ctx=ctx,
                    requested_model=model,
                    info=info,
                    status=getattr(e, "status_code", 502),
                    started=started,
                    cache_hit=False,
                    stream=False,
                    error=str(e),
                )
                raise
            cache_hit = note_cache(ctx) or cache_hit
            spawn(
                record(
                    key=key,
                    ctx=ctx,
                    requested_model=model,
                    info=info,
                    status=200,
                    started=started,
                    cache_hit=cache_hit,
                    stream=False,
                    usage=resp.usage,
                )
            )
            return JSONResponse(
                resp.model_dump(mode="json", exclude_none=True),
                headers={
                    "x-gateway-provider": info.provider or "cache",
                    "x-gateway-fallbacks": str(info.fallbacks),
                },
            )

        # ---- streaming ----
        try:
            req2, short, ran = await pipeline.run_before(req, ctx)
            if short is None:
                gen = router.stream(req2, info)
                first = await anext(gen, None)  # surfaces pre-first-chunk failures as HTTP errors
            else:
                gen, first = None, None
        except (UnknownModel, ProviderError, AllProvidersFailed) as e:
            await record(
                key=key,
                ctx=ctx,
                requested_model=model,
                info=info,
                status=getattr(e, "status_code", 502),
                started=started,
                cache_hit=False,
                stream=True,
                error=str(e),
            )
            raise

        # PII placeholders must be restored in-flight for the client, while the aggregated
        # response handed to after() (and thus the cache) keeps the placeholders.
        restorer = next(
            (r for mw in ran if (f := getattr(mw, "stream_restorer", None)) and (r := f(ctx))), None
        )

        def render(c: ChatChunk, final: bool = False) -> list[bytes]:
            if restorer is None:
                return [_sse_line(c)]
            out: list[ChatChunk] = []
            choices = [ch for ch in c.choices if ch.index == 0]
            if choices and choices[0].delta.content:
                restored = restorer.feed(choices[0].delta.content)
                c = c.model_copy(
                    update={
                        "choices": [
                            ch.model_copy(
                                update={"delta": ch.delta.model_copy(update={"content": restored})}
                            )
                            if ch.index == 0
                            else ch
                            for ch in c.choices
                        ]
                    }
                )
            if final or any(ch.finish_reason for ch in c.choices):
                tail = restorer.flush()
                if tail:
                    out.append(
                        c.model_copy(
                            update={
                                "choices": [ChatChunkChoice(delta=ChatDelta(content=tail))],
                                "usage": None,
                            }
                        )
                    )
            out.append(c)
            return [_sse_line(x) for x in out]

        async def sse() -> AsyncGenerator[bytes, None]:
            chunks: list[ChatChunk] = []
            err: str | None = None
            cache_hit = short is not None
            try:
                if short is not None:
                    for c in response_to_chunks(short):
                        chunks.append(c)
                        for b in render(c):
                            yield b
                else:
                    assert gen is not None
                    if first is not None:
                        chunks.append(first)
                        for b in render(first):
                            yield b
                    async for c in gen:
                        chunks.append(c)
                        for b in render(c):
                            yield b
            except (ProviderError, AllProvidersFailed) as e:
                err = str(e)
                if restorer is not None and (tail := restorer.flush()):
                    yield _sse_line(
                        ChatChunk(
                            id="chatcmpl-tail",
                            created=int(time.time()),
                            model=model,
                            choices=[ChatChunkChoice(delta=ChatDelta(content=tail))],
                        )
                    )
                payload = {"error": {"message": err, "type": "upstream_error"}}
                yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            usage = short.usage if short is not None else info.usage
            full = aggregate_chunks(chunks, info.upstream_model or req2.model, usage)
            if err is None:
                try:
                    await pipeline.run_after(req2, full, ctx, ran)
                except Exception:
                    log.exception("after() failed on streamed response")
            hit = note_cache(ctx) or cache_hit
            await record(
                key=key,
                ctx=ctx,
                requested_model=model,
                info=info,
                status=200 if err is None else 502,
                started=started,
                cache_hit=hit,
                stream=True,
                error=err,
                usage=usage,
            )

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "x-gateway-provider": info.provider or "cache"},
        )

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        key = await authenticate(request)
        if isinstance(key, JSONResponse):
            return key
        router: Router = app.state.router
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "llm-gateway"}
                    for m in router.models()
                ],
            }
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        try:
            await app.state.redis.ping()
            async with app.state.sm() as s:
                await s.execute(text("SELECT 1"))
        except Exception as e:
            return _err(503, f"not ready: {e}", "not_ready")
        return JSONResponse({"status": "ready"})

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):  # several uvicorn workers: aggregate them
            reg = CollectorRegistry()
            multiprocess.MultiProcessCollector(reg)  # type: ignore[no-untyped-call]
            return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)
        return Response(generate_latest(app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
