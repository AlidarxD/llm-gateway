"""Default middleware pipeline. Order: pii -> cache -> provider."""

from __future__ import annotations

from typing import Any

from gateway.middleware.base import Middleware


def default_middlewares(settings: Any, redis: Any) -> list[Middleware]:
    """Wired up once Agent B's pii/cache packages are merged."""
    return []
