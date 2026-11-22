"""Metric hooks. The core can pass its own `CacheMetrics` (e.g. Prometheus-backed)."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Protocol

Kind = Literal["exact", "semantic"]
Outcome = Literal["hit", "miss", "skip", "error"]


class CacheMetrics(Protocol):
    def on_lookup(self, kind: Kind, outcome: Outcome, latency_seconds: float) -> None: ...
    def on_store(self, kind: Kind, latency_seconds: float) -> None: ...


class InMemoryCacheMetrics:
    """Reference implementation; also handy in tests."""

    def __init__(self) -> None:
        self.lookups: Counter[tuple[str, str]] = Counter()
        self.stores: Counter[str] = Counter()
        self.lookup_latency_sum: dict[str, float] = {}

    def on_lookup(self, kind: Kind, outcome: Outcome, latency_seconds: float) -> None:
        self.lookups[(kind, outcome)] += 1
        self.lookup_latency_sum[kind] = self.lookup_latency_sum.get(kind, 0.0) + latency_seconds

    def on_store(self, kind: Kind, latency_seconds: float) -> None:
        self.stores[kind] += 1

    def hit_rate(self) -> float:
        hits = sum(v for (_, o), v in self.lookups.items() if o == "hit")
        total = sum(v for (_, o), v in self.lookups.items() if o in ("hit", "miss"))
        return hits / total if total else 0.0
