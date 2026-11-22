"""Prometheus metrics."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "gateway_requests_total", "Client requests", ["status", "model"], registry=r
        )
        self.provider_latency = Histogram(
            "gateway_provider_latency_seconds",
            "Provider call latency",
            ["provider", "outcome"],
            buckets=BUCKETS,
            registry=r,
        )
        self.request_latency = Histogram(
            "gateway_request_latency_seconds", "End-to-end latency", buckets=BUCKETS, registry=r
        )
        self.tokens = Counter(
            "gateway_tokens_total", "Tokens", ["provider", "model", "type"], registry=r
        )
        self.cost = Counter(
            "gateway_cost_usd_total", "Cost in USD", ["provider", "model"], registry=r
        )
        self.fallbacks = Counter(
            "gateway_fallbacks_total", "Fallbacks", ["from_provider", "to_provider"], registry=r
        )
        self.retries = Counter("gateway_retries_total", "Retries", ["provider"], registry=r)
        self.breaker_state = Gauge(
            "gateway_circuit_breaker_open", "1 if breaker open", ["provider"], registry=r
        )
        self.cache = Counter(
            "gateway_cache_requests_total", "Cache lookups", ["result"], registry=r
        )
        self.cache_latency = Histogram(
            "gateway_cache_lookup_seconds",
            "Cache lookup latency",
            ["kind", "outcome"],
            buckets=BUCKETS,
            registry=r,
        )
        self.rate_limited = Counter(
            "gateway_rate_limited_total", "Rejected by limits", ["reason"], registry=r
        )
