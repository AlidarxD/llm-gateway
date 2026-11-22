"""End-to-end: default pipeline (pii -> cache -> provider) through the HTTP API."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

import fakeredis
import httpx
import pytest
from sqlalchemy import select

from gateway.app import create_app
from gateway.auth import create_key
from gateway.config import Settings
from gateway.db import UsageLog
from gateway.providers.fake import FakeProvider
from gateway.schemas import ChatChunk, ChatChunkChoice, ChatDelta, ChatRequest
from gateway.wiring import default_middlewares

PHONE = "+7 701 123 45 67"
BODY = {"model": "smart", "messages": [{"role": "user", "content": f"call me at {PHONE}"}]}


class Spy(FakeProvider):
    """Records what the provider actually received; streams in 3-char pieces so that PII
    placeholders get split across chunks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.seen: list[ChatRequest] = []

    async def complete(self, req, model):  # type: ignore[no-untyped-def]
        self.seen.append(req)
        return await super().complete(req, model)

    async def stream(self, req, model) -> AsyncGenerator[ChatChunk, None]:  # type: ignore[no-untyped-def]
        self.seen.append(req)
        await self._maybe_fail(req)
        reply, cid = self._reply(req, model), f"chatcmpl-{uuid.uuid4().hex}"
        for i in range(0, len(reply), 3):
            yield ChatChunk(
                id=cid,
                created=int(time.time()),
                model=model,
                choices=[ChatChunkChoice(delta=ChatDelta(content=reply[i : i + 3]))],
            )
        yield ChatChunk(
            id=cid,
            created=int(time.time()),
            model=model,
            choices=[ChatChunkChoice(delta=ChatDelta(), finish_reason="stop")],
        )


async def _make(settings, sm, routes, metrics, monkeypatch, **overrides):
    for k, v in overrides.items():
        setattr(settings, k, v)
    monkeypatch.setattr(Settings, "load_routes", lambda self: routes)
    redis = fakeredis.FakeAsyncRedis()
    provs = {"primary": Spy("primary"), "secondary": Spy("secondary")}

    async def nosleep(_: float) -> None:
        return None

    app = create_app(
        settings,
        redis=redis,
        session_maker=sm,
        providers=provs,
        middlewares=default_middlewares(settings, redis, metrics),
        metrics=metrics,
        sleep=nosleep,
    )
    return app, redis, provs


@pytest.fixture
async def env(settings, sm, routes, metrics, monkeypatch):
    app, redis, provs = await _make(settings, sm, routes, metrics, monkeypatch)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            yield c, app, redis, provs


async def _stream_text(c, body, headers):
    async with c.stream(
        "POST", "/v1/chat/completions", json={**body, "stream": True}, headers=headers
    ) as r:
        provider = r.headers["x-gateway-provider"]
        lines = [ln async for ln in r.aiter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    return "".join(
        ch["delta"].get("content") or "" for c_ in chunks for ch in c_.get("choices", [])
    ), provider


async def _flush(app):
    while app.state.tasks:
        await asyncio.gather(*list(app.state.tasks))
    await app.state.usage.flush()


async def test_pii_redacted_upstream_and_restored_for_client(env, auth):
    c, app, _, provs = env
    r = await c.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r.status_code == 200
    sent = str(provs["primary"].seen[0].messages[0].content)
    assert PHONE not in sent and "<PHONE_1>" in sent
    assert PHONE in r.json()["choices"][0]["message"]["content"]


async def test_streaming_restores_placeholders_split_across_chunks(env, auth):
    c, _, _, provs = env
    text, _ = await _stream_text(c, BODY, auth)
    assert PHONE in text and "<PHONE" not in text
    assert PHONE not in str(provs["primary"].seen[0].messages[0].content)


async def test_exact_cache_hit_skips_provider_and_costs_nothing(env, auth, sm):
    c, app, _, provs = env
    body = {"model": "smart", "messages": [{"role": "user", "content": "what is 2+2?"}]}
    r1 = await c.post("/v1/chat/completions", json=body, headers=auth)
    r2 = await c.post("/v1/chat/completions", json=body, headers=auth)
    assert r1.headers["x-gateway-provider"] == "primary"
    assert r2.headers["x-gateway-provider"] == "cache"
    assert len(provs["primary"].seen) == 1
    assert r1.json()["choices"] == r2.json()["choices"]
    await _flush(app)
    async with sm() as s:
        rows = (await s.execute(select(UsageLog).order_by(UsageLog.created_at))).scalars().all()
    assert sorted(r.cache_hit for r in rows) == [False, True]
    assert [r.cost_usd for r in rows if r.cache_hit] == [0]
    assert 'gateway_cache_requests_total{result="hit"} 1.0' in (await c.get("/metrics")).text


async def test_cached_pii_response_restored_per_request_and_stored_redacted(env, auth):
    c, _, redis, provs = env
    r1 = await c.post("/v1/chat/completions", json=BODY, headers=auth)
    r2 = await c.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r2.headers["x-gateway-provider"] == "cache"
    assert PHONE in r2.json()["choices"][0]["message"]["content"]
    assert len(provs["primary"].seen) == 1
    for k in await redis.keys("*"):
        assert PHONE.encode() not in (await redis.get(k) or b"")  # never stored in clear text
    assert r1.status_code == 200


async def test_stream_then_non_stream_share_cache_and_hit_is_replayed_as_stream(env, auth):
    c, _, _, provs = env
    body = {"model": "smart", "messages": [{"role": "user", "content": "tell me a joke"}]}
    text1, p1 = await _stream_text(c, body, auth)
    r = await c.post("/v1/chat/completions", json=body, headers=auth)
    assert p1 == "primary" and r.headers["x-gateway-provider"] == "cache"
    assert r.json()["choices"][0]["message"]["content"].strip() == text1.strip()
    text3, p3 = await _stream_text(c, body, auth)
    assert p3 == "cache" and text3.strip() == text1.strip()
    assert len(provs["primary"].seen) == 1


async def test_tenants_do_not_share_cache(env, sm):
    c, _, _, provs = env
    k1, _ = await create_key(sm, "tenant-a", "a", rpm_limit=100)
    k2, _ = await create_key(sm, "tenant-b", "b", rpm_limit=100)
    body = {"model": "smart", "messages": [{"role": "user", "content": "secret strategy?"}]}
    for k in (k1, k2):
        r = await c.post(
            "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {k}"}
        )
        assert r.headers["x-gateway-provider"] == "primary"
    assert len(provs["primary"].seen) == 2


async def test_high_temperature_is_never_cached(env, auth):
    c, _, _, provs = env
    body = {"model": "smart", "temperature": 0.9, "messages": [{"role": "user", "content": "poem"}]}
    for _ in range(2):
        await c.post("/v1/chat/completions", json=body, headers=auth)
    assert len(provs["primary"].seen) == 2


async def test_pii_block_mode_short_circuits(settings, sm, routes, metrics, monkeypatch, auth):
    app, _, provs = await _make(
        settings, sm, routes, metrics, monkeypatch, pii_default_mode="block"
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r = await c.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "content_filter"
    assert provs["primary"].seen == []


async def test_failed_provider_response_is_not_cached(env, auth):
    c, _, _, provs = env
    provs["primary"].script.extend(["500"] * 3)
    body = {"model": "smart", "messages": [{"role": "user", "content": "flaky question"}]}
    r1 = await c.post("/v1/chat/completions", json=body, headers=auth)  # falls back to secondary
    assert r1.headers["x-gateway-provider"] == "secondary"
    r2 = await c.post("/v1/chat/completions", json=body, headers=auth)
    assert r2.headers["x-gateway-provider"] == "cache"
