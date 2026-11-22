"""Runs middlewares in order around a provider call (or around a streamed call)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from gateway.middleware.base import Middleware
from gateway.schemas import ChatRequest, ChatResponse, RequestContext

ProviderCall = Callable[[ChatRequest, RequestContext], Awaitable[ChatResponse]]


class Pipeline:
    def __init__(self, middlewares: Sequence[Middleware]) -> None:
        self.middlewares = list(middlewares)

    async def run_before(
        self, req: ChatRequest, ctx: RequestContext
    ) -> tuple[ChatRequest, ChatResponse | None, list[Middleware]]:
        """before() in order. Returns (request, short-circuit response or None, middlewares that ran)."""
        ran: list[Middleware] = []
        for mw in self.middlewares:
            ran.append(mw)
            out = await mw.before(req, ctx)
            if isinstance(out, ChatResponse):
                return req, out, ran
            req = out
        return req, None, ran

    async def run_after(
        self, req: ChatRequest, resp: ChatResponse, ctx: RequestContext, ran: list[Middleware]
    ) -> ChatResponse:
        """after() in reverse order, only for middlewares whose before() ran."""
        for mw in reversed(ran):
            resp = await mw.after(req, resp, ctx)
        return resp

    async def run(
        self, req: ChatRequest, ctx: RequestContext, provider: ProviderCall
    ) -> ChatResponse:
        req, resp, ran = await self.run_before(req, ctx)
        if resp is None:
            resp = await provider(req, ctx)
        return await self.run_after(req, resp, ctx, ran)
