"""Admin CLI: `python -m gateway.cli create-key --tenant acme --name dev`."""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal

from gateway.auth import create_key
from gateway.config import Settings
from gateway.db import make_engine, make_sessionmaker


async def _create(a: argparse.Namespace) -> None:
    engine = make_engine(Settings().database_url)
    raw, row = await create_key(
        make_sessionmaker(engine),
        a.tenant,
        a.name,
        a.rpm,
        a.daily_tokens,
        Decimal(a.daily_usd) if a.daily_usd is not None else None,
    )
    await engine.dispose()
    print(f"key id: {row.id}\nAPI key (shown once): {raw}")


def main() -> None:
    p = argparse.ArgumentParser(prog="gateway.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-key")
    c.add_argument("--tenant", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--rpm", type=int, default=60)
    c.add_argument("--daily-tokens", type=int)
    c.add_argument("--daily-usd")
    c.set_defaults(fn=_create)
    a = p.parse_args()
    asyncio.run(a.fn(a))


if __name__ == "__main__":
    main()
