"""Settings and model-alias routing tables."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Target(BaseModel):
    provider: str
    model: str


# Dev profile: no paid calls (fake + Ollama only).
DEV_ROUTES: dict[str, list[Target]] = {
    "smart": [
        Target(provider="fake", model="fake-smart"),
        Target(provider="ollama", model="llama3.2"),
    ],
    "fast": [
        Target(provider="fake", model="fake-fast"),
        Target(provider="ollama", model="llama3.2"),
    ],
}

# Full profile: needs real provider keys.
FULL_ROUTES: dict[str, list[Target]] = {
    "smart": [
        Target(provider="anthropic", model="claude-sonnet-5-5"),
        Target(provider="openrouter", model="anthropic/claude-sonnet-4.5"),
    ],
    "fast": [
        Target(provider="ollama", model="llama3.2"),
        Target(provider="openai", model="gpt-4o-mini"),
    ],
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=None, extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    redis_url: str = "redis://localhost:6379/0"

    enabled_providers: list[str] = ["fake", "ollama"]
    routes_file: str | None = None

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    request_timeout_s: float = 30.0
    max_retries: int = 2
    backoff_base_s: float = 0.2
    backoff_max_s: float = 2.0
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0

    auth_cache_ttl_s: float = 30.0
    log_level: str = "INFO"

    def load_routes(self) -> dict[str, list[Target]]:
        if self.routes_file:
            raw = json.loads(Path(self.routes_file).read_text())
            return {k: [Target(**t) for t in v] for k, v in raw.items()}
        return DEV_ROUTES
