"""Batched usage-log writer: requests enqueue rows, a background task bulk-inserts them."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.db import UsageLog

log = logging.getLogger("gateway.usage")


class UsageWriter:
    def __init__(
        self,
        sm: async_sessionmaker[AsyncSession],
        interval_s: float = 0.5,
        batch_size: int = 200,
    ) -> None:
        self.sm = sm
        self.interval_s = interval_s
        self.batch_size = batch_size
        self._queue: asyncio.Queue[UsageLog] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def add(self, row: UsageLog) -> None:
        self._queue.put_nowait(row)

    async def flush(self) -> None:
        while not self._queue.empty():
            batch: list[UsageLog] = []
            while len(batch) < self.batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                async with self.sm() as s:
                    s.add_all(batch)
                    await s.commit()
            except Exception:
                log.exception("usage batch insert failed", extra={"rows": len(batch)})

    async def _run(self) -> None:
        while not self._stop.is_set():
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), self.interval_s)
            await self.flush()

    async def close(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
        await self.flush()
