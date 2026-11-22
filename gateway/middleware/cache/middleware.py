"""Cache middleware: exact + semantic. Placed AFTER the PII middleware (pii -> cache)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from gateway.middleware.cache import keys
from gateway.middleware.cache.embedder import Embedder
from gateway.middleware.cache.guard import signature
from gateway.middleware.cache.metrics import CacheMetrics, Kind, Outcome
from gateway.middleware.cache.store import ExactStore, RedisLike, SemanticIndex
from gateway.schemas import ChatRequest, ChatResponse, RequestContext

logger = logging.getLogger("gateway.cache")

STATE_KEY = "_cache_state"
DEFAULT_SEMANTIC_THRESHOLD = 0.94  # chosen from benchmarks/cache_eval.py, see docs/cache.md


@dataclass(slots=True)
class CacheConfig:
    ttl_seconds: int = 3600
    # Requests with temperature above this are never cached or served from cache.
    max_temperature: float = 0.3
    # Opt-in: on adversarial minimal pairs the semantic cache false-hits often (docs/cache.md).
    semantic_enabled: bool = False
    default_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    # Similarity threshold per model alias; falls back to default_threshold.
    thresholds: Mapping[str, float] = field(default_factory=dict)
    max_semantic_entries: int = 2000
    # Require equal numbers/negation between query and cached text for a semantic hit.
    lexical_guard: bool = True

    def threshold_for(self, model: str) -> float:
        return self.thresholds.get(model, self.default_threshold)


@dataclass(slots=True)
class _State:
    exact_key: str
    store: bool = True
    semantic: tuple[str, Any, str] | None = None  # (context_hash, vector, signature)


class CacheMiddleware:
    """`before` returns a ChatResponse on a hit; `after` stores the provider result.

    Sets ctx.extra["cache_hit"] (bool), ["cache_type"] ("exact"|"semantic"|None) and
    ["cache_similarity"] on semantic hits. Cache failures (Redis down, embedder error)
    degrade to a miss and never fail the request.
    """

    name = "cache"

    def __init__(
        self,
        redis: RedisLike,
        embedder: Embedder | None = None,
        config: CacheConfig | None = None,
        metrics: CacheMetrics | None = None,
    ) -> None:
        self.config = config or CacheConfig()
        self.embedder = embedder
        self.metrics = metrics
        self._exact = ExactStore(redis, self.config.ttl_seconds)
        self._semantic = SemanticIndex(
            redis, self.config.ttl_seconds, self.config.max_semantic_entries
        )

    def _observe(self, kind: Kind, outcome: Outcome, started: float) -> None:
        if self.metrics is not None:
            self.metrics.on_lookup(kind, outcome, time.perf_counter() - started)

    async def before(self, req: ChatRequest, ctx: RequestContext) -> ChatRequest | ChatResponse:
        ctx.extra["cache_hit"] = False
        ctx.extra["cache_type"] = None
        if not keys.request_cacheable(req, self.config.max_temperature):
            ctx.extra.pop(STATE_KEY, None)
            return req
        state = _State(exact_key=keys.exact_key(req))
        ctx.extra[STATE_KEY] = state

        started = time.perf_counter()
        try:
            raw = await self._exact.get(ctx.tenant_id, state.exact_key)
        except Exception:
            logger.warning("cache exact lookup failed", exc_info=True)
            self._observe("exact", "error", started)
            raw = None
        else:
            self._observe("exact", "hit" if raw else "miss", started)
        if raw:
            return self._hit(raw, ctx, "exact")

        if not (self.config.semantic_enabled and self.embedder is not None):
            return req
        split = keys.split_for_semantic(req)
        # Placeholders like <IIN_1> are request-local; a paraphrase's numbering may differ,
        # so semantic reuse is disabled for redacted requests (exact cache still applies).
        if split is None or keys.has_placeholders(split[1]):
            self._observe("semantic", "skip", time.perf_counter())
            return req
        context_hash, text = split
        sig = signature(text) if self.config.lexical_guard else ""
        started = time.perf_counter()
        try:
            vec = (await asyncio.to_thread(self.embedder.embed, [text]))[0]
            state.semantic = (context_hash, vec, sig)
            hit = await self._semantic.search(
                ctx.tenant_id,
                req.model,
                context_hash,
                vec,
                self.config.threshold_for(req.model),
                sig,
            )
        except Exception:
            logger.warning("cache semantic lookup failed", exc_info=True)
            self._observe("semantic", "error", started)
            return req
        self._observe("semantic", "hit" if hit else "miss", started)
        if hit:
            ctx.extra["cache_similarity"] = hit.similarity
            return self._hit(hit.response_json, ctx, "semantic")
        return req

    def _hit(self, raw: str, ctx: RequestContext, kind: Kind) -> ChatResponse:
        ctx.extra["cache_hit"] = True
        ctx.extra["cache_type"] = kind
        ctx.extra.pop(STATE_KEY, None)  # nothing to store on a hit
        return ChatResponse.model_validate_json(raw)

    async def after(
        self, req: ChatRequest, resp: ChatResponse, ctx: RequestContext
    ) -> ChatResponse:
        state = ctx.extra.pop(STATE_KEY, None)
        if not isinstance(state, _State) or ctx.extra.get("cache_hit"):
            return resp
        if not keys.response_cacheable(resp):
            return resp
        raw = resp.model_dump_json()
        try:
            started = time.perf_counter()
            await self._exact.put(ctx.tenant_id, state.exact_key, raw)
            if self.metrics is not None:
                self.metrics.on_store("exact", time.perf_counter() - started)
            if state.semantic is not None:
                started = time.perf_counter()
                context_hash, vec, sig = state.semantic
                await self._semantic.add(ctx.tenant_id, req.model, context_hash, vec, raw, sig)
                if self.metrics is not None:
                    self.metrics.on_store("semantic", time.perf_counter() - started)
        except Exception:
            logger.warning("cache store failed", exc_info=True)
        return resp
