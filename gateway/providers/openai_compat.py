"""Adapter for OpenAI-compatible APIs: OpenAI, OpenRouter, Ollama (/v1)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx

from gateway.errors import ProviderServerError, ProviderTimeout
from gateway.providers.base import map_status
from gateway.schemas import ChatChunk, ChatRequest, ChatResponse


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.headers = dict(extra_headers or {})
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _payload(self, req: ChatRequest, model: str, stream: bool) -> dict[str, object]:
        body = req.model_dump(exclude_none=True)
        body["model"] = model
        body["stream"] = stream
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, req: ChatRequest, model: str) -> ChatResponse:
        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(req, model, False),
                headers=self.headers,
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeout(self.name, str(e)) from e
        except httpx.TransportError as e:
            raise ProviderServerError(self.name, f"transport: {e}") from e
        if r.status_code >= 400:
            raise map_status(self.name, r.status_code, r.text, r.headers.get("retry-after"))
        return ChatResponse.model_validate(r.json())

    async def stream(self, req: ChatRequest, model: str) -> AsyncGenerator[ChatChunk, None]:
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=self._payload(req, model, True),
                headers=self.headers,
            ) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    raise map_status(self.name, r.status_code, body, r.headers.get("retry-after"))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    if data:
                        yield ChatChunk.model_validate_json(data)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(self.name, str(e)) from e
        except httpx.TransportError as e:
            raise ProviderServerError(self.name, f"transport: {e}") from e

    async def list_models(self) -> list[str]:
        try:
            r = await self.client.get(f"{self.base_url}/models", headers=self.headers)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return []
