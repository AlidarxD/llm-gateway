from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal

import fakeredis
import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from gateway.app import create_app
from gateway.auth import create_key
from gateway.config import Settings, Target
from gateway.db import Base, make_sessionmaker
from gateway.metrics import Metrics
from gateway.providers.fake import FakeProvider


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _prices(monkeypatch):
    from gateway import costs

    monkeypatch.setitem(costs.PRICES, "p-model", (Decimal("3"), Decimal("15")))
    monkeypatch.setitem(costs.PRICES, "s-model", (Decimal("1"), Decimal("2")))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        max_retries=2, breaker_failure_threshold=3, breaker_reset_s=30, request_timeout_s=2
    )


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


@pytest.fixture
async def sm():
    engine = create_async_engine(
        "sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    yield make_sessionmaker(engine)
    await engine.dispose()


@pytest.fixture
def providers() -> dict[str, FakeProvider]:
    return {"primary": FakeProvider("primary"), "secondary": FakeProvider("secondary")}


@pytest.fixture
def routes() -> dict[str, list[Target]]:
    return {
        "smart": [
            Target(provider="primary", model="p-model"),
            Target(provider="secondary", model="s-model"),
        ]
    }


@pytest.fixture
async def client(
    settings, sm, providers, routes, metrics, monkeypatch
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr(Settings, "load_routes", lambda self: routes)
    redis = fakeredis.FakeAsyncRedis()
    app = create_app(
        settings,
        redis=redis,
        session_maker=sm,
        providers=providers,
        middlewares=[],
        metrics=metrics,
        sleep=_no_sleep,
    )
    async with app.router.lifespan_context(app):
        app.state.test_redis = redis
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            c.app = app  # type: ignore[attr-defined]
            yield c


@pytest.fixture
async def api_key(sm) -> str:
    raw, _ = await create_key(sm, "acme", "test", rpm_limit=1000)
    return raw


@pytest.fixture
def auth(api_key) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def usd() -> type[Decimal]:
    return Decimal
