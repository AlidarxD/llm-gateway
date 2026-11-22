"""Per-model cost table (USD per 1M tokens). Unknown models cost 0 and are flagged by the caller."""

from __future__ import annotations

from decimal import Decimal

from gateway.schemas import Usage

# model -> (input $/1M, output $/1M)
PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "claude-sonnet-5-5": (Decimal("3"), Decimal("15")),
    "claude-haiku-5-5": (Decimal("1"), Decimal("5")),
    "anthropic/claude-sonnet-4.5": (Decimal("3"), Decimal("15")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "gpt-4o": (Decimal("2.50"), Decimal("10")),
    "llama3.2": (Decimal("0"), Decimal("0")),
    "fake-smart": (Decimal("3"), Decimal("15")),  # priced so budget logic is demoable
    "fake-fast": (Decimal("0.15"), Decimal("0.60")),
}

MILLION = Decimal(1_000_000)


def is_priced(model: str) -> bool:
    return model in PRICES


def compute_cost(model: str, usage: Usage) -> Decimal:
    p_in, p_out = PRICES.get(model, (Decimal(0), Decimal(0)))
    return (p_in * usage.prompt_tokens + p_out * usage.completion_tokens) / MILLION
