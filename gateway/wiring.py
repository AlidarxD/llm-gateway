"""Default middleware pipeline. Order: pii -> cache -> provider."""

from __future__ import annotations

from typing import Any

from gateway.config import Settings
from gateway.metrics import Metrics
from gateway.middleware.base import Middleware
from gateway.middleware.cache import CacheConfig, CacheMiddleware
from gateway.middleware.pii import PIIConfig, PIIMiddleware, PIIMode


class PromCacheMetrics:
    """Adapts the cache middleware's metrics hooks to Prometheus. Hit/miss counts come from
    ctx.extra["cache_hit"] in the app, so only latency is recorded here."""

    def __init__(self, metrics: Metrics) -> None:
        self.m = metrics

    def on_lookup(self, kind: str, outcome: str, latency_seconds: float) -> None:
        self.m.cache_latency.labels(kind, outcome).observe(latency_seconds)

    def on_store(self, kind: str, latency_seconds: float) -> None:
        self.m.cache_latency.labels(kind, "store").observe(latency_seconds)


def default_middlewares(settings: Settings, redis: Any, metrics: Metrics) -> list[Middleware]:
    mws: list[Middleware] = []
    if settings.pii_enabled:
        mws.append(PIIMiddleware(PIIConfig(default_mode=PIIMode(settings.pii_default_mode))))
    if settings.cache_enabled:
        mws.append(
            CacheMiddleware(
                redis,
                config=CacheConfig(
                    ttl_seconds=settings.cache_ttl_s, max_temperature=settings.cache_max_temperature
                ),
                metrics=PromCacheMetrics(metrics),
            )
        )
    return mws
