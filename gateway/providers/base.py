"""Provider adapter interface and shared HTTP error mapping."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from gateway.errors import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderServerError,
)
from gateway.schemas import ChatChunk, ChatRequest, ChatResponse


class Provider(Protocol):
    name: str

    async def complete(self, req: ChatRequest, model: str) -> ChatResponse: ...

    def stream(self, req: ChatRequest, model: str) -> AsyncGenerator[ChatChunk, None]: ...

    async def list_models(self) -> list[str]: ...


def map_status(provider: str, status: int, body: str, retry_after: str | None) -> ProviderError:
    snippet = body[:200]
    if status == 429:
        try:
            ra = float(retry_after) if retry_after else None
        except ValueError:
            ra = None
        return ProviderRateLimited(provider, snippet, retry_after=ra)
    if status in (401, 403):
        return ProviderAuthError(provider, snippet, status=status)
    if status >= 500 or status in (408, 409):
        return ProviderServerError(provider, snippet, status=status)
    return ProviderBadRequest(provider, snippet, status=status)
