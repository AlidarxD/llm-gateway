"""Runs middlewares in order around a provider call."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from gateway.middleware.base import Middleware
from gateway.schemas import ChatRequest, ChatResponse, RequestContext

ProviderCall = Callable[[ChatRequest, RequestContext], Awaitable[ChatResponse]]


class Pipeline:
    def __init__(self, middlewares: Sequence[Middleware]) -> None:
        self.middlewares = list(middlewares)

    async def run(
        self, req: ChatRequest, ctx: RequestContext, provider: ProviderCall
    ) -> ChatResponse:
        """before() in order; on short-circuit, after() runs only for middlewares
        that already ran before() (including the one that short-circuited)."""
        ran: list[Middleware] = []
        resp: ChatResponse | None = None
        for mw in self.middlewares:
            ran.append(mw)
            out = await mw.before(req, ctx)
            if isinstance(out, ChatResponse):
                resp = out
                break
            req = out
        if resp is None:
            resp = await provider(req, ctx)
        for mw in reversed(ran):
            resp = await mw.after(req, resp, ctx)
        return resp
