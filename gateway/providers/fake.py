"""Deterministic fake provider for tests and demos, with failure injection.

Failure injection:
  * `FakeProvider.script`: a queue of outcomes consumed one per call ("ok", "timeout", "429", "500", "400").
  * Prompt triggers: a user message containing `[fake:timeout]`, `[fake:429]`, `[fake:500]` or `[fake:slow]`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator

from gateway.errors import (
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from gateway.providers.tokens import estimate_tokens
from gateway.schemas import (
    ChatChoice,
    ChatChunk,
    ChatChunkChoice,
    ChatDelta,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Usage,
)


class FakeProvider:
    def __init__(self, name: str = "fake", latency_s: float = 0.0) -> None:
        self.name = name
        self.latency_s = latency_s
        self.script: deque[str] = deque()
        self.calls = 0

    def _last_user(self, req: ChatRequest) -> str:
        for m in reversed(req.messages):
            if m.role == "user":
                c = m.content
                return c if isinstance(c, str) else str(c)
        return ""

    async def _maybe_fail(self, req: ChatRequest) -> None:
        self.calls += 1
        text = self._last_user(req)
        outcome = self.script.popleft() if self.script else "ok"
        for trig in ("timeout", "429", "500", "400", "slow"):
            if f"[fake:{trig}]" in text:
                outcome = trig
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if outcome == "timeout":
            raise ProviderTimeout(self.name, "simulated timeout")
        if outcome == "429":
            raise ProviderRateLimited(self.name, "simulated 429", retry_after=0.0)
        if outcome == "500":
            raise ProviderServerError(self.name, "simulated 500", status=500)
        if outcome == "400":
            raise ProviderBadRequest(self.name, "simulated 400", status=400)
        if outcome == "slow":
            await asyncio.sleep(3600)

    def _reply(self, req: ChatRequest, model: str) -> str:
        return f"[{model}] echo: {self._last_user(req)}"

    async def complete(self, req: ChatRequest, model: str) -> ChatResponse:
        await self._maybe_fail(req)
        reply = self._reply(req, model)
        pt = sum(estimate_tokens(str(m.content)) for m in req.messages)
        ct = estimate_tokens(reply)
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=reply), finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
        )

    async def stream(self, req: ChatRequest, model: str) -> AsyncGenerator[ChatChunk, None]:
        await self._maybe_fail(req)
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        reply = self._reply(req, model)
        pt = sum(estimate_tokens(str(m.content)) for m in req.messages)

        def mk(
            delta: ChatDelta, finish: str | None = None, usage: Usage | None = None
        ) -> ChatChunk:
            return ChatChunk(
                id=cid,
                created=created,
                model=model,
                choices=[ChatChunkChoice(delta=delta, finish_reason=finish)],
                usage=usage,
            )

        yield mk(ChatDelta(role="assistant", content=""))
        for word in reply.split(" "):
            yield mk(ChatDelta(content=word + " "))
        ct = estimate_tokens(reply)
        yield mk(
            ChatDelta(), "stop", Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        )

    async def list_models(self) -> list[str]:
        return ["fake-smart", "fake-fast"]
