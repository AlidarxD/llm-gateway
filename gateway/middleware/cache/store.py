"""Redis-backed stores: exact (hash -> response) and semantic (fallback brute-force index)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class RedisLike(Protocol):
    """The subset of redis.asyncio.Redis we use (also satisfied by fakeredis)."""

    async def get(self, name: str) -> Any: ...
    async def set(self, name: str, value: Any, ex: int | None = None) -> Any: ...
    async def mget(self, keys: list[str]) -> Any: ...
    async def zadd(self, name: str, mapping: dict[str, float]) -> Any: ...
    async def zrevrange(self, name: str, start: int, end: int) -> Any: ...
    async def zrem(self, name: str, *values: str) -> Any: ...
    async def zcard(self, name: str) -> Any: ...
    async def zremrangebyrank(self, name: str, start: int, end: int) -> Any: ...
    async def expire(self, name: str, time: int) -> Any: ...


PREFIX = "gw:cache"


def exact_redis_key(tenant: str, key: str) -> str:
    return f"{PREFIX}:x:{tenant}:{key}"


class ExactStore:
    def __init__(self, redis: RedisLike, ttl_seconds: int) -> None:
        self._r = redis
        self._ttl = ttl_seconds

    async def get(self, tenant: str, key: str) -> str | None:
        raw = await self._r.get(exact_redis_key(tenant, key))
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    async def put(self, tenant: str, key: str, response_json: str) -> None:
        await self._r.set(exact_redis_key(tenant, key), response_json, ex=self._ttl)


@dataclass(frozen=True, slots=True)
class SemanticHit:
    similarity: float
    response_json: str


class SemanticIndex:
    """Per (tenant, model) index. Fallback implementation without RediSearch:

    entries live under `gw:cache:s:{tenant}:{model}:{id}` as JSON
    {ctx, vec(list[float]), resp} with a TTL; ids are tracked in a sorted set
    (score = insert time) capped at `max_entries`; lookup does an exact brute-force
    cosine over the entries whose context hash matches. Tenant and model are part of
    every key, so entries from other tenants are never even loaded.
    """

    def __init__(self, redis: RedisLike, ttl_seconds: int, max_entries: int = 2000) -> None:
        self._r = redis
        self._ttl = ttl_seconds
        self._max = max_entries

    def _idx(self, tenant: str, model: str) -> str:
        return f"{PREFIX}:sidx:{tenant}:{model}"

    def _key(self, tenant: str, model: str, eid: str) -> str:
        return f"{PREFIX}:s:{tenant}:{model}:{eid}"

    async def search(
        self,
        tenant: str,
        model: str,
        context_hash: str,
        vec: NDArray[np.float32],
        threshold: float,
        sig: str = "",
    ) -> SemanticHit | None:
        ids = [
            i.decode() if isinstance(i, bytes) else str(i)
            for i in await self._r.zrevrange(self._idx(tenant, model), 0, self._max - 1)
        ]
        if not ids:
            return None
        raws = await self._r.mget([self._key(tenant, model, i) for i in ids])
        best: SemanticHit | None = None
        stale: list[str] = []
        for eid, raw in zip(ids, raws, strict=True):
            if raw is None:
                stale.append(eid)  # TTL expired; drop from the index
                continue
            entry = json.loads(raw)
            if entry["ctx"] != context_hash or entry.get("sig", "") != sig:
                continue
            sim = float(np.dot(vec, np.asarray(entry["vec"], dtype=np.float32)))
            if sim >= threshold and (best is None or sim > best.similarity):
                best = SemanticHit(sim, entry["resp"])
        if stale:
            await self._r.zrem(self._idx(tenant, model), *stale)
        return best

    async def add(
        self,
        tenant: str,
        model: str,
        context_hash: str,
        vec: NDArray[np.float32],
        response_json: str,
        sig: str = "",
    ) -> None:
        eid = f"{time.time_ns():x}"
        payload = json.dumps(
            {
                "ctx": context_hash,
                "sig": sig,
                "vec": [round(float(x), 6) for x in vec],
                "resp": response_json,
            }
        )
        await self._r.set(self._key(tenant, model, eid), payload, ex=self._ttl)
        idx = self._idx(tenant, model)
        await self._r.zadd(idx, {eid: time.time()})
        await self._r.expire(idx, self._ttl)
        size = int(await self._r.zcard(idx))
        if size > self._max:
            await self._r.zremrangebyrank(idx, 0, size - self._max - 1)
