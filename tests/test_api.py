import json

from sqlalchemy import select

from gateway.db import UsageLog

BODY = {"model": "smart", "messages": [{"role": "user", "content": "hi there"}]}


async def test_requires_auth(client):
    assert (await client.post("/v1/chat/completions", json=BODY)).status_code == 401
    r = await client.post(
        "/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"}
    )
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


async def test_chat_completion(client, auth, sm):
    r = await client.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["choices"][0]["message"]["content"].endswith("hi there")
    assert d["usage"]["total_tokens"] > 0
    assert r.headers["x-gateway-provider"] == "primary"
    assert "x-request-id" in r.headers
    await _drain(client)
    async with sm() as s:
        rows = (await s.execute(select(UsageLog))).scalars().all()
    assert len(rows) == 1 and rows[0].status == 200 and rows[0].provider == "primary"
    assert rows[0].cost_usd >= 0


async def _drain(client):
    import asyncio

    while client.app.state.tasks:  # type: ignore[attr-defined]
        await asyncio.gather(*list(client.app.state.tasks))


async def test_fallback_visible_in_headers(client, auth, providers):
    providers["primary"].script.extend(["500"] * 3)
    r = await client.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "secondary"
    assert r.headers["x-gateway-fallbacks"] == "1"


async def test_all_fail_returns_502_openai_error(client, auth, providers):
    providers["primary"].script.extend(["500"] * 3)
    providers["secondary"].script.extend(["500"] * 3)
    r = await client.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r.status_code == 502 and r.json()["error"]["code"] == "all_providers_failed"


async def test_bad_request_maps_to_400(client, auth, providers):
    providers["primary"].script.append("400")
    r = await client.post("/v1/chat/completions", json=BODY, headers=auth)
    assert r.status_code == 400


async def test_unknown_model_404(client, auth):
    r = await client.post("/v1/chat/completions", json={**BODY, "model": "zzz"}, headers=auth)
    assert r.status_code == 404


async def test_streaming_sse(client, auth):
    async with client.stream(
        "POST", "/v1/chat/completions", json={**BODY, "stream": True}, headers=auth
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln async for ln in r.aiter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert "hi there" in text


async def test_streaming_falls_back(client, auth, providers):
    providers["primary"].script.extend(["429"] * 3)
    async with client.stream(
        "POST", "/v1/chat/completions", json={**BODY, "stream": True}, headers=auth
    ) as r:
        assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"
        await r.aread()


async def test_models_list(client, auth):
    r = await client.get("/v1/models", headers=auth)
    assert r.status_code == 200 and r.json()["data"][0]["id"] == "smart"


async def test_rate_limit(client, sm):
    from gateway.auth import create_key

    raw, _ = await create_key(sm, "t", "limited", rpm_limit=2)
    h = {"Authorization": f"Bearer {raw}"}
    codes = [
        (await client.post("/v1/chat/completions", json=BODY, headers=h)).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
    r = await client.post("/v1/chat/completions", json=BODY, headers=h)
    assert "retry-after" in r.headers


async def test_token_budget_exhausted(client, sm):
    from gateway.auth import create_key

    raw, _ = await create_key(sm, "t", "budget", rpm_limit=100, daily_token_budget=5)
    h = {"Authorization": f"Bearer {raw}"}
    assert (await client.post("/v1/chat/completions", json=BODY, headers=h)).status_code == 200
    await _drain(client)
    r = await client.post("/v1/chat/completions", json=BODY, headers=h)
    assert r.status_code == 429 and r.json()["error"]["code"] == "token_budget"


async def test_usd_budget_exhausted(client, sm):
    from decimal import Decimal

    from gateway.auth import create_key

    raw, _ = await create_key(sm, "t", "usd", rpm_limit=100, daily_usd_budget=Decimal("0.000001"))
    h = {"Authorization": f"Bearer {raw}"}
    assert (await client.post("/v1/chat/completions", json=BODY, headers=h)).status_code == 200
    await _drain(client)
    r = await client.post("/v1/chat/completions", json=BODY, headers=h)
    assert r.status_code == 429 and r.json()["error"]["code"] == "usd_budget"


async def test_metrics_endpoint(client, auth):
    await client.post("/v1/chat/completions", json=BODY, headers=auth)
    await _drain(client)
    text = (await client.get("/metrics")).text
    assert "gateway_requests_total" in text and "gateway_provider_latency_seconds_bucket" in text
    assert "gateway_cost_usd_total" in text


async def test_health(client):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    assert (await client.get("/readyz")).status_code == 200


async def test_fake_failure_rate_is_probabilistic():
    import random

    from gateway.errors import ProviderError
    from gateway.providers.fake import FakeProvider
    from gateway.schemas import ChatMessage, ChatRequest

    random.seed(1)
    p = FakeProvider(failure_rate=0.5)
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="x")])
    ok = fail = 0
    for _ in range(100):
        try:
            await p.complete(req, "m")
            ok += 1
        except ProviderError:
            fail += 1
    assert 20 < fail < 80 and ok + fail == 100
