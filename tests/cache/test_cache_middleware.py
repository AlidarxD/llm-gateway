import hashlib
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from fakeredis import FakeAsyncRedis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pii"))

from pii_dataset import make_iin  # noqa: E402

from gateway.middleware.cache import CacheConfig, CacheMiddleware, InMemoryCacheMetrics
from gateway.middleware.cache.embedder import Vector
from gateway.middleware.pii import PIIMiddleware
from gateway.pipeline import Pipeline
from gateway.schemas import ChatChoice, ChatMessage, ChatRequest, ChatResponse, RequestContext


class FakeEmbedder:
    """Texts sharing a key in `groups` get near-identical vectors; others are unrelated."""

    dim = 16

    def __init__(self, groups: dict[str, str] | None = None) -> None:
        self.groups = groups or {}

    def embed(self, texts: Sequence[str]) -> Vector:
        out = []
        for t in texts:
            key = self.groups.get(t, t)
            seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            if t in self.groups:  # tiny per-text jitter keeps sim < 1 but very high
                j = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
                v = v + 0.02 * j.normal(size=self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def ctx(tenant: str = "t1") -> RequestContext:
    return RequestContext(api_key_id="k", tenant_id=tenant, request_id="r")


def req(text: str, **kw: object) -> ChatRequest:
    return ChatRequest(model="m", messages=[ChatMessage(role="user", content=text)], **kw)  # type: ignore[arg-type]


def resp(text: str = "answer", finish: str = "stop") -> ChatResponse:
    return ChatResponse(
        id="x",
        created=1,
        model="m",
        choices=[
            ChatChoice(message=ChatMessage(role="assistant", content=text), finish_reason=finish)
        ],
    )


class Provider:
    def __init__(self, text: str = "answer", finish: str = "stop") -> None:
        self.calls = 0
        self.text, self.finish = text, finish

    async def __call__(self, r: ChatRequest, c: RequestContext) -> ChatResponse:
        self.calls += 1
        return resp(self.text, self.finish)


def make(
    embedder: FakeEmbedder | None = None, **cfg: object
) -> tuple[CacheMiddleware, FakeAsyncRedis]:
    r = FakeAsyncRedis()
    return CacheMiddleware(r, embedder, CacheConfig(**cfg)), r  # type: ignore[arg-type]


async def run(
    mw: CacheMiddleware, r: ChatRequest, p: Provider, c: RequestContext | None = None
) -> RequestContext:
    c = c or ctx()
    await Pipeline([mw]).run(r, c, p)
    return c


async def test_exact_hit_and_normalization() -> None:
    mw, _ = make()
    p = Provider()
    c1 = await run(mw, req("Hello   world"), p)
    c2 = await run(mw, req("  Hello world "), p)
    assert p.calls == 1 and not c1.extra["cache_hit"] and c2.extra["cache_hit"]
    assert c2.extra["cache_type"] == "exact"


async def test_different_params_or_model_miss() -> None:
    mw, _ = make()
    p = Provider()
    await run(mw, req("hi", max_tokens=10), p)
    await run(mw, req("hi", max_tokens=20), p)
    await run(mw, ChatRequest(model="other", messages=[ChatMessage(role="user", content="hi")]), p)
    assert p.calls == 3


async def test_not_cached_with_high_temperature_tools_or_bad_finish() -> None:
    mw, r = make(max_temperature=0.3)
    p = Provider()
    for _ in range(2):
        await run(mw, req("a", temperature=0.9), p)
    assert p.calls == 2
    tools = req("b", tools=[{"type": "function", "function": {"name": "f"}}])
    for _ in range(2):
        await run(mw, tools, p)
    assert p.calls == 4
    p2 = Provider(finish="length")
    for _ in range(2):
        await run(mw, req("c"), p2)
    assert p2.calls == 2
    assert await r.dbsize() == 0


async def test_low_temperature_is_cached() -> None:
    mw, _ = make(max_temperature=0.3)
    p = Provider()
    await run(mw, req("a", temperature=0.2), p)
    await run(mw, req("a", temperature=0.2), p)
    assert p.calls == 1


async def test_tenant_isolation_exact_and_semantic() -> None:
    emb = FakeEmbedder({"how to pay tax": "g", "how do I pay taxes": "g"})
    mw, _ = make(emb, default_threshold=0.9)
    p = Provider()
    await run(mw, req("how to pay tax"), p, ctx("A"))
    c = await run(mw, req("how to pay tax"), p, ctx("B"))
    assert not c.extra["cache_hit"] and p.calls == 2
    c = await run(mw, req("how do I pay taxes"), p, ctx("C"))
    assert not c.extra["cache_hit"] and p.calls == 3
    c = await run(mw, req("how do I pay taxes"), p, ctx("A"))
    assert c.extra["cache_type"] == "semantic" and p.calls == 3


async def test_semantic_hit_threshold_and_per_model_override() -> None:
    emb = FakeEmbedder({"q one": "g", "q uno": "g"})
    mw, _ = make(emb, default_threshold=0.9)
    p = Provider()
    await run(mw, req("q one"), p)
    c = await run(mw, req("q uno"), p)
    assert c.extra["cache_type"] == "semantic" and c.extra["cache_similarity"] > 0.9
    strict, _ = make(emb, default_threshold=0.9, thresholds={"m": 0.9999})
    p2 = Provider()
    await run(strict, req("q one"), p2)
    c = await run(strict, req("q uno"), p2)
    assert not c.extra["cache_hit"] and p2.calls == 2


async def test_semantic_requires_same_history_and_params() -> None:
    emb = FakeEmbedder({"q one": "g", "q uno": "g"})
    mw, _ = make(emb, default_threshold=0.9)
    p = Provider()
    sys_a = ChatMessage(role="system", content="You are A")
    sys_b = ChatMessage(role="system", content="You are B")
    await run(
        mw, ChatRequest(model="m", messages=[sys_a, ChatMessage(role="user", content="q one")]), p
    )
    r2 = ChatRequest(model="m", messages=[sys_b, ChatMessage(role="user", content="q uno")])
    c = await run(mw, r2, p)
    assert not c.extra["cache_hit"]


async def test_semantic_skipped_for_redacted_placeholders() -> None:
    emb = FakeEmbedder({"iin <IIN_1> ok": "g", "iin <IIN_1> fine": "g"})
    mw, _ = make(emb, default_threshold=0.9)
    p = Provider()
    await run(mw, req("iin <IIN_1> ok"), p)
    c = await run(mw, req("iin <IIN_1> fine"), p)
    assert not c.extra["cache_hit"]


async def test_cache_after_pii_never_stores_pii() -> None:
    iin = make_iin(random.Random(3))
    r = FakeAsyncRedis()
    cache = CacheMiddleware(r, FakeEmbedder(), CacheConfig())
    pipe = Pipeline([PIIMiddleware(), cache])
    seen: list[str] = []

    async def provider(q: ChatRequest, c: RequestContext) -> ChatResponse:
        seen.append(str(q.messages[0].content))
        return resp(f"checked {q.messages[0].content}")

    out1 = await pipe.run(req(f"check iin {iin}"), ctx(), provider)
    c2 = ctx()
    out2 = await pipe.run(req(f"check iin {iin}"), c2, provider)
    assert len(seen) == 1 and c2.extra["cache_hit"]
    assert iin in str(out1.choices[0].message.content) and iin in str(
        out2.choices[0].message.content
    )
    for k in await r.keys("*"):
        blob = str(await r.get(k)) + k.decode()
        assert iin not in blob


async def test_redis_failure_degrades_to_miss() -> None:
    class Broken(FakeAsyncRedis):
        async def get(self, *a: object, **k: object) -> object:
            raise ConnectionError("down")

        async def set(self, *a: object, **k: object) -> object:
            raise ConnectionError("down")

    mw = CacheMiddleware(Broken(), None, CacheConfig())
    p = Provider()
    await run(mw, req("x"), p)
    await run(mw, req("x"), p)
    assert p.calls == 2


async def test_metrics_hooks() -> None:
    m = InMemoryCacheMetrics()
    r = FakeAsyncRedis()
    mw = CacheMiddleware(r, None, CacheConfig(), metrics=m)
    p = Provider()
    await run(mw, req("x"), p)
    await run(mw, req("x"), p)
    assert m.lookups[("exact", "miss")] == 1 and m.lookups[("exact", "hit")] == 1
    assert m.stores["exact"] == 1 and m.hit_rate() == pytest.approx(0.5)


async def test_semantic_index_cap_and_ttl_cleanup() -> None:
    mw, r = make(FakeEmbedder(), max_semantic_entries=3)
    p = Provider()
    for i in range(6):
        await run(mw, req(f"unique question {i}"), p)
    assert await r.zcard("gw:cache:sidx:t1:m") == 3
