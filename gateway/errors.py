"""Provider and gateway errors."""

from __future__ import annotations


class ProviderError(Exception):
    """Base provider failure. `retryable` means retry/fallback makes sense."""

    retryable = True
    status_code = 502
    code = "upstream_error"

    def __init__(self, provider: str, message: str = "", *, status: int | None = None) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message
        self.status = status


class ProviderTimeout(ProviderError):
    status_code = 504
    code = "upstream_timeout"


class ProviderRateLimited(ProviderError):
    code = "upstream_rate_limited"

    def __init__(
        self, provider: str, message: str = "", *, retry_after: float | None = None
    ) -> None:
        super().__init__(provider, message, status=429)
        self.retry_after = retry_after


class ProviderServerError(ProviderError):
    code = "upstream_server_error"


class ProviderBadRequest(ProviderError):
    """4xx other than 429: the request itself is wrong; do not retry or fall back."""

    retryable = False
    status_code = 400
    code = "invalid_request"


class ProviderAuthError(ProviderError):
    """Our upstream credentials are wrong: skip this provider but it is not the client's fault."""

    code = "upstream_auth_error"


class AllProvidersFailed(Exception):
    status_code = 502
    code = "all_providers_failed"

    def __init__(self, errors: list[ProviderError]) -> None:
        super().__init__("; ".join(str(e) for e in errors) or "no available provider")
        self.errors = errors


class UnknownModel(Exception):
    status_code = 404
    code = "model_not_found"
