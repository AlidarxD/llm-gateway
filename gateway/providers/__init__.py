"""Provider registry."""

from __future__ import annotations

import httpx

from gateway.config import Settings
from gateway.providers.anthropic import AnthropicProvider
from gateway.providers.base import Provider
from gateway.providers.fake import FakeProvider
from gateway.providers.openai_compat import OpenAICompatProvider


def build_providers(settings: Settings, client: httpx.AsyncClient) -> dict[str, Provider]:
    out: dict[str, Provider] = {}
    for name in settings.enabled_providers:
        if name == "fake":
            out[name] = FakeProvider()
        elif name == "ollama":
            out[name] = OpenAICompatProvider("ollama", f"{settings.ollama_base_url}/v1", client)
        elif name == "openai" and settings.openai_api_key:
            out[name] = OpenAICompatProvider(
                "openai",
                "https://api.openai.com/v1",
                client,
                settings.openai_api_key.get_secret_value(),
            )
        elif name == "openrouter" and settings.openrouter_api_key:
            out[name] = OpenAICompatProvider(
                "openrouter",
                "https://openrouter.ai/api/v1",
                client,
                settings.openrouter_api_key.get_secret_value(),
            )
        elif name == "anthropic" and settings.anthropic_api_key:
            out[name] = AnthropicProvider(client, settings.anthropic_api_key.get_secret_value())
    return out
