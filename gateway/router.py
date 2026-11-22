"""Model-alias routing with timeouts, retries, per-provider circuit breakers and fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal

from gateway.config import Settings, Target
from gateway.costs import compute_cost
from gateway.errors import (
    AllProvidersFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    UnknownModel,
)
from gateway.metrics import Metrics
from gateway.providers.base import Provider
from gateway.providers.tokens import estimate_tokens
from gateway.reliability import BreakerState, CircuitBreaker, backoff_delay
from gateway.schemas import ChatChunk, ChatRequest, ChatResponse, Usage

log = logging.getLogger("gateway.router")


@dataclass
class RouteInfo:
    """What actually served a request; filled in by the router."""

    provider: str = ""
    upstream_model: str = ""
    fallbacks: int = 0
    retries: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: Decimal = Decimal(0)
    errors: list[str] = field(default_factory=list)


class Router:
    def __init__(
        self,
        providers: dict[str, Provider],
        routes: dict[str, list[Target]],
        settings: Settings,
        metrics: Metrics,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = providers
        self.routes = routes
        self.s = settings
        self.m = metrics
        self._sleep = sleep
        self.breakers = {
            n: CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_reset_s, clock)
            for n in providers
        }

    # ---- model resolution -------------------------------------------------
    def models(self) -> list[str]:
        return sorted(self.routes)

    def resolve(self, model: str) -> list[Target]:
        chain = self.routes.get(model)
        if chain is None and "/" in model:  # "provider/model" passthrough
            prov, _, up = model.partition("/")
            if prov in self.providers:
                chain = [Target(provider=prov, model=up)]
        if not chain:
            raise UnknownModel(f"model '{model}' not found")
        chain = [t for t in chain if t.provider in self.providers]
        if not chain:
            raise UnknownModel(f"no enabled provider for model '{model}'")
        return chain

    # ---- shared helpers ---------------------------------------------------
    def _breaker_gauge(self, name: str) -> None:
        self.m.breaker_state.labels(name).set(
            1 if self.breakers[name].state is BreakerState.OPEN else 0
        )

    def _record_failure(self, name: str, err: ProviderError, started: float) -> None:
        outcome = "timeout" if isinstance(err, ProviderTimeout) else "error"
        self.m.provider_latency.labels(name, outcome).observe(time.monotonic() - started)
        if not isinstance(err, ProviderBadRequest):
            self.breakers[name].record_failure()
        self._breaker_gauge(name)

    def _finish(self, info: RouteInfo, usage: Usage) -> None:
        info.usage = usage
        info.cost_usd = compute_cost(info.upstream_model, usage)
        self.m.tokens.labels(info.provider, info.upstream_model, "prompt").inc(usage.prompt_tokens)
        self.m.tokens.labels(info.provider, info.upstream_model, "completion").inc(
            usage.completion_tokens
        )
        self.m.cost.labels(info.provider, info.upstream_model).inc(float(info.cost_usd))

    async def _retry_wait(self, attempt: int, err: ProviderError) -> None:
        delay = backoff_delay(attempt, self.s.backoff_base_s, self.s.backoff_max_s)
        if isinstance(err, ProviderRateLimited) and err.retry_after:
            delay = max(delay, min(err.retry_after, self.s.backoff_max_s))
        await self._sleep(delay)

    # ---- non-streaming ----------------------------------------------------
    async def complete(self, req: ChatRequest, info: RouteInfo) -> ChatResponse:
        chain = self.resolve(req.model)
        errors: list[ProviderError] = []
        prev: str | None = None
        for target in chain:
            name, provider, br = (
                target.provider,
                self.providers[target.provider],
                self.breakers[target.provider],
            )
            if prev is not None:
                self.m.fallbacks.labels(prev, name).inc()
                info.fallbacks += 1
            prev = name
            attempts = self.s.max_retries + 1
            for attempt in range(attempts):
                if not br.allow():
                    errors.append(ProviderError(name, "circuit open"))
                    self._breaker_gauge(name)
                    break
                started = time.monotonic()
                try:
                    async with asyncio.timeout(self.s.request_timeout_s):
                        resp = await provider.complete(req, target.model)
                except TimeoutError:
                    err: ProviderError = ProviderTimeout(name, "timeout")
                except ProviderError as e:
                    err = e
                else:
                    br.record_success()
                    self._breaker_gauge(name)
                    self.m.provider_latency.labels(name, "ok").observe(time.monotonic() - started)
                    info.provider, info.upstream_model = name, target.model
                    self._finish(info, resp.usage)
                    return resp
                self._record_failure(name, err, started)
                info.errors.append(str(err))
                if not err.retryable:
                    raise err
                errors.append(err)
                if attempt < attempts - 1:
                    info.retries += 1
                    self.m.retries.labels(name).inc()
                    await self._retry_wait(attempt, err)
            log.warning("provider exhausted", extra={"provider": name, "model": req.model})
        raise AllProvidersFailed(errors)

    # ---- streaming --------------------------------------------------------
    async def stream(self, req: ChatRequest, info: RouteInfo) -> AsyncIterator[ChatChunk]:
        """Retries/fallback apply until the first chunk arrives; after that we are committed."""
        chain = self.resolve(req.model)
        errors: list[ProviderError] = []
        prev: str | None = None
        for target in chain:
            name, provider, br = (
                target.provider,
                self.providers[target.provider],
                self.breakers[target.provider],
            )
            if prev is not None:
                self.m.fallbacks.labels(prev, name).inc()
                info.fallbacks += 1
            prev = name
            attempts = self.s.max_retries + 1
            for attempt in range(attempts):
                if not br.allow():
                    errors.append(ProviderError(name, "circuit open"))
                    self._breaker_gauge(name)
                    break
                started = time.monotonic()
                gen = provider.stream(req, target.model)
                try:
                    async with asyncio.timeout(self.s.request_timeout_s):
                        first = await anext(gen)
                except StopAsyncIteration:
                    first = None
                except TimeoutError:
                    await gen.aclose()
                    err: ProviderError = ProviderTimeout(name, "timeout waiting for first chunk")
                    self._record_failure(name, err, started)
                    info.errors.append(str(err))
                    errors.append(err)
                    if attempt < attempts - 1:
                        info.retries += 1
                        self.m.retries.labels(name).inc()
                        await self._retry_wait(attempt, err)
                    continue
                except ProviderError as e:
                    self._record_failure(name, e, started)
                    info.errors.append(str(e))
                    if not e.retryable:
                        raise
                    errors.append(e)
                    if attempt < attempts - 1:
                        info.retries += 1
                        self.m.retries.labels(name).inc()
                        await self._retry_wait(attempt, e)
                    continue
                # committed to this provider
                info.provider, info.upstream_model = name, target.model
                text_len = 0
                usage: Usage | None = None
                try:
                    chunks = [] if first is None else [first]
                    for c in chunks:
                        text_len += sum(len(ch.delta.content or "") for ch in c.choices)
                        usage = c.usage or usage
                        yield c
                    async for c in gen:
                        text_len += sum(len(ch.delta.content or "") for ch in c.choices)
                        usage = c.usage or usage
                        yield c
                except ProviderError as e:
                    self._record_failure(name, e, started)
                    raise
                br.record_success()
                self._breaker_gauge(name)
                self.m.provider_latency.labels(name, "ok").observe(time.monotonic() - started)
                if usage is None:
                    pt = sum(estimate_tokens(str(m.content)) for m in req.messages)
                    ct = max(1, text_len // 4)
                    usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
                self._finish(info, usage)
                return
            log.warning("provider exhausted (stream)", extra={"provider": name})
        raise AllProvidersFailed(errors)
