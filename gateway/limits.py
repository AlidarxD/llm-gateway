"""Redis-backed per-key rate limit (fixed window) and daily token/USD budgets."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from redis.asyncio import Redis

from gateway.auth import KeyInfo
from gateway.schemas import BudgetInfo

MICRO = Decimal(1_000_000)


@dataclass
class LimitDecision:
    allowed: bool
    reason: str | None = None  # "rate_limit" | "token_budget" | "usd_budget"
    retry_after_s: int = 0
    budget: BudgetInfo | None = None


def _day(now: float) -> str:
    return time.strftime("%Y%m%d", time.gmtime(now))


class Limiter:
    def __init__(self, redis: Redis, clock: Callable[[], float] = time.time) -> None:
        self.r = redis
        self._clock = clock

    def _now(self) -> float:
        return self._clock()

    async def check(self, key: KeyInfo) -> LimitDecision:
        now = self._now()
        window = int(now // 60)
        rk = f"rl:{key.id}:{window}"
        count = await self.r.incr(rk)
        if count == 1:
            await self.r.expire(rk, 65)
        if count > key.rpm_limit:
            return LimitDecision(False, "rate_limit", retry_after_s=int(60 - now % 60) + 1)

        day = _day(now)
        tokens = int(await self.r.get(f"budget:{key.id}:{day}:tokens") or 0)
        micro = int(await self.r.get(f"budget:{key.id}:{day}:usd_micro") or 0)
        tok_rem = None if key.daily_token_budget is None else key.daily_token_budget - tokens
        usd_rem = (
            None if key.daily_usd_budget is None else key.daily_usd_budget - Decimal(micro) / MICRO
        )
        budget = BudgetInfo(tokens_remaining=tok_rem, usd_remaining=usd_rem)
        reset = int(86400 - now % 86400) + 1
        if tok_rem is not None and tok_rem <= 0:
            return LimitDecision(False, "token_budget", reset, budget)
        if usd_rem is not None and usd_rem <= 0:
            return LimitDecision(False, "usd_budget", reset, budget)
        return LimitDecision(True, budget=budget)

    async def record(self, key: KeyInfo, tokens: int, cost_usd: Decimal) -> None:
        day = _day(self._now())
        tk, uk = f"budget:{key.id}:{day}:tokens", f"budget:{key.id}:{day}:usd_micro"
        async with self.r.pipeline(transaction=False) as p:
            p.incrby(tk, tokens)
            p.expire(tk, 172800)
            p.incrby(uk, int((cost_usd * MICRO).to_integral_value()))
            p.expire(uk, 172800)
            await p.execute()
