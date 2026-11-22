"""Integration tests: fake provider simulating timeouts, 429s and 500s."""

import asyncio

import pytest

from gateway.errors import AllProvidersFailed, ProviderBadRequest
from gateway.metrics import Metrics
from gateway.reliability import BreakerState
from gateway.router import RouteInfo, Router
from gateway.schemas import ChatMessage, ChatRequest

REQ = ChatRequest(model="smart", messages=[ChatMessage(role="user", content="hello")])


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def router(providers, routes, settings, metrics):
    return Router(providers, routes, settings, metrics, sleep=_no_sleep)


async def test_retry_then_success_on_same_provider(router, providers):
    providers["primary"].script.extend(["500", "429"])
    info = RouteInfo()
    resp = await router.complete(REQ, info)
    assert info.provider == "primary" and info.retries == 2 and info.fallbacks == 0
    assert providers["primary"].calls == 3
    assert resp.model == "p-model"


async def test_fallback_after_retries_exhausted(router, providers, metrics):
    providers["primary"].script.extend(["500"] * 3)
    info = RouteInfo()
    resp = await router.complete(REQ, info)
    assert info.provider == "secondary" and info.fallbacks == 1
    assert resp.model == "s-model"
    assert metrics.fallbacks.labels("primary", "secondary")._value.get() == 1


@pytest.mark.parametrize("failure", ["timeout", "429", "500"])
async def test_each_failure_type_triggers_fallback(router, providers, failure):
    providers["primary"].script.extend([failure] * 3)
    info = RouteInfo()
    await router.complete(REQ, info)
    assert info.provider == "secondary"


async def test_real_asyncio_timeout_falls_back(providers, routes, settings, metrics):
    settings.request_timeout_s = 0.05
    settings.max_retries = 0
    r = Router(providers, routes, settings, metrics, sleep=_no_sleep)
    req = ChatRequest(model="smart", messages=[ChatMessage(role="user", content="[fake:slow]")])
    # the [fake:slow] trigger makes every provider hang, so the router timeout fires on each
    with pytest.raises(AllProvidersFailed):
        await asyncio.wait_for(r.complete(req, RouteInfo()), 5)
    assert providers["primary"].calls == 1 and providers["secondary"].calls == 1


async def test_circuit_opens_and_skips_provider(router, providers):
    providers["primary"].script.extend(["500"] * 100)
    for _ in range(2):  # 3 failures per request -> breaker (threshold 3) opens in first request
        await router.complete(REQ, RouteInfo())
    assert router.breakers["primary"].state is BreakerState.OPEN
    calls = providers["primary"].calls
    info = RouteInfo()
    await router.complete(REQ, info)
    assert providers["primary"].calls == calls  # skipped, not called
    assert info.provider == "secondary"


async def test_circuit_half_open_recovers(providers, routes, settings, metrics):
    t = [0.0]
    r = Router(providers, routes, settings, metrics, sleep=_no_sleep, clock=lambda: t[0])
    providers["primary"].script.extend(["500"] * 3)
    await r.complete(REQ, RouteInfo())
    assert r.breakers["primary"].state is BreakerState.OPEN
    t[0] = 31
    info = RouteInfo()
    await r.complete(REQ, info)
    assert info.provider == "primary"
    assert r.breakers["primary"].state is BreakerState.CLOSED


async def test_bad_request_is_not_retried_or_failed_over(router, providers):
    providers["primary"].script.append("400")
    with pytest.raises(ProviderBadRequest):
        await router.complete(REQ, RouteInfo())
    assert providers["primary"].calls == 1 and providers["secondary"].calls == 0
    assert router.breakers["primary"].state is BreakerState.CLOSED


async def test_all_providers_failed(router, providers):
    providers["primary"].script.extend(["500"] * 3)
    providers["secondary"].script.extend(["timeout"] * 3)
    with pytest.raises(AllProvidersFailed) as e:
        await router.complete(REQ, RouteInfo())
    assert len(e.value.errors) == 6


async def test_stream_fallback_before_first_chunk(router, providers):
    providers["primary"].script.extend(["500"] * 3)
    info = RouteInfo()
    chunks = [c async for c in router.stream(REQ, info)]
    text = "".join(ch.delta.content or "" for c in chunks for ch in c.choices)
    assert "s-model" in text and info.provider == "secondary" and info.usage.total_tokens > 0


def test_unknown_model(router):
    from gateway.errors import UnknownModel

    with pytest.raises(UnknownModel):
        router.resolve("nope")


def test_provider_slash_passthrough(router):
    assert router.resolve("primary/xyz")[0].model == "xyz"


def test_metrics_isolated():
    assert Metrics().registry is not Metrics().registry
