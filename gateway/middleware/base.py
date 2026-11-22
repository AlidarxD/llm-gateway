"""Middleware contract. See CONTRACT.md. Do not change without updating CONTRACT.md."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gateway.schemas import ChatRequest, ChatResponse, RequestContext


@runtime_checkable
class Middleware(Protocol):
    name: str

    async def before(self, req: ChatRequest, ctx: RequestContext) -> ChatRequest | ChatResponse:
        """Return a (possibly modified) request, or a ChatResponse to short-circuit."""
        ...

    async def after(
        self, req: ChatRequest, resp: ChatResponse, ctx: RequestContext
    ) -> ChatResponse:
        """Return a (possibly modified) response."""
        ...
