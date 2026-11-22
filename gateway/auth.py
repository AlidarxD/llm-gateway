"""API keys: generation, hashed lookup (with a short TTL cache)."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.db import ApiKey

PREFIX = "sk-gw-"


def hash_key(raw: str) -> str:
    # Keys are 256-bit random, so a fast unsalted SHA-256 is sufficient (see ADR 0003).
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_key() -> str:
    return PREFIX + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class KeyInfo:
    id: str
    tenant_id: str
    rpm_limit: int
    daily_token_budget: int | None
    daily_usd_budget: Decimal | None


async def create_key(
    sm: async_sessionmaker[AsyncSession],
    tenant_id: str,
    name: str,
    rpm_limit: int = 60,
    daily_token_budget: int | None = None,
    daily_usd_budget: Decimal | None = None,
) -> tuple[str, ApiKey]:
    raw = generate_key()
    row = ApiKey(
        tenant_id=tenant_id,
        name=name,
        key_hash=hash_key(raw),
        key_prefix=raw[:10],
        rpm_limit=rpm_limit,
        daily_token_budget=daily_token_budget,
        daily_usd_budget=daily_usd_budget,
    )
    async with sm() as s:
        s.add(row)
        await s.commit()
    return raw, row


class KeyStore:
    def __init__(self, sm: async_sessionmaker[AsyncSession], ttl_s: float = 30.0) -> None:
        self.sm = sm
        self.ttl = ttl_s
        self._cache: dict[str, tuple[float, KeyInfo | None]] = {}

    async def lookup(self, raw: str) -> KeyInfo | None:
        h = hash_key(raw)
        hit = self._cache.get(h)
        now = time.monotonic()
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        async with self.sm() as s:
            row = (await s.execute(select(ApiKey).where(ApiKey.key_hash == h))).scalar_one_or_none()
        info = (
            KeyInfo(
                row.id, row.tenant_id, row.rpm_limit, row.daily_token_budget, row.daily_usd_budget
            )
            if row and row.is_active
            else None
        )
        self._cache[h] = (now, info)
        return info
