"""Adapter for the native Anthropic Messages API (translated to/from OpenAI schema)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from gateway.errors import ProviderServerError, ProviderTimeout
from gateway.providers.base import map_status
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

STOP_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        version: str = "2023-06-01",
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-api-key": api_key, "anthropic-version": version}

    def _payload(self, req: ChatRequest, model: str, stream: bool) -> dict[str, Any]:
        system = "\n".join(
            _text(m.content) for m in req.messages if m.role in ("system", "developer")
        )
        messages = [
            {"role": m.role, "content": _text(m.content)}
            for m in req.messages
            if m.role in ("user", "assistant")
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens or 1024,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if req.temperature is not None:
            body["temperature"] = min(req.temperature, 1.0)
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
        return body

    async def complete(self, req: ChatRequest, model: str) -> ChatResponse:
        try:
            r = await self.client.post(
                f"{self.base_url}/v1/messages",
                json=self._payload(req, model, False),
                headers=self.headers,
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeout(self.name, str(e)) from e
        except httpx.TransportError as e:
            raise ProviderServerError(self.name, f"transport: {e}") from e
        if r.status_code >= 400:
            raise map_status(self.name, r.status_code, r.text, r.headers.get("retry-after"))
        d = r.json()
        usage = d.get("usage", {})
        pt, ct = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return ChatResponse(
            id=d.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            created=int(time.time()),
            model=d.get("model", model),
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=_text(d.get("content", []))),
                    finish_reason=STOP_MAP.get(d.get("stop_reason") or "", "stop"),
                )
            ],
            usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
        )

    async def stream(self, req: ChatRequest, model: str) -> AsyncGenerator[ChatChunk, None]:
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        prompt_tokens = 0

        def chunk(
            delta: ChatDelta, finish: str | None = None, usage: Usage | None = None
        ) -> ChatChunk:
            return ChatChunk(
                id=cid,
                created=created,
                model=model,
                choices=[ChatChunkChoice(delta=delta, finish_reason=finish)],
                usage=usage,
            )

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=self._payload(req, model, True),
                headers=self.headers,
            ) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    raise map_status(self.name, r.status_code, body, r.headers.get("retry-after"))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:])
                    t = ev.get("type")
                    if t == "message_start":
                        prompt_tokens = ev["message"].get("usage", {}).get("input_tokens", 0)
                        yield chunk(ChatDelta(role="assistant", content=""))
                    elif t == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                        yield chunk(ChatDelta(content=ev["delta"]["text"]))
                    elif t == "message_delta":
                        ct = ev.get("usage", {}).get("output_tokens", 0)
                        stop = STOP_MAP.get(ev["delta"].get("stop_reason") or "", "stop")
                        yield chunk(
                            ChatDelta(),
                            stop,
                            Usage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=ct,
                                total_tokens=prompt_tokens + ct,
                            ),
                        )
                    elif t == "error":
                        raise ProviderServerError(self.name, str(ev.get("error")))
        except httpx.TimeoutException as e:
            raise ProviderTimeout(self.name, str(e)) from e
        except httpx.TransportError as e:
            raise ProviderServerError(self.name, f"transport: {e}") from e

    async def list_models(self) -> list[str]:
        return []
