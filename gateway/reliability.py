"""Circuit breaker and jittered-backoff helpers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Consecutive-failure breaker. After `reset_s` in OPEN, lets one probe through (HALF_OPEN)."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_s = reset_s
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state = BreakerState.CLOSED
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self.reset_s:
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allow(self) -> bool:
        s = self.state
        if s is BreakerState.CLOSED:
            return True
        if s is BreakerState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._probe_in_flight = False
        self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self._probe_in_flight = False
        if self.state is BreakerState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()


def backoff_delay(
    attempt: int, base: float, cap: float, rng: Callable[[], float] = random.random
) -> float:
    """Full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return float(rng() * min(cap, base * (2**attempt)))
