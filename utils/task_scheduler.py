import asyncio
import logging
from collections import defaultdict
from contextlib import suppress
from typing import Any, Awaitable


class TaskScheduler:
    """Simple centralized scheduler with a limit on active tasks."""

    def __init__(self, max_active: int = 5):
        self.max_active = max_active
        self._queue: asyncio.Queue[tuple[Awaitable[Any], str | None]] = asyncio.Queue()
        self._active: set[asyncio.Task] = set()
        self._user_tasks: defaultdict[str, set[asyncio.Task]] = defaultdict(set)
        self._logger = logging.getLogger(__name__)
        loop = asyncio.get_event_loop()
        self._runner = loop.create_task(self._run())

    async def _run(self) -> None:
        while True:
            coro, user_id = await self._queue.get()
            await self._start_task(coro, user_id)

    async def _start_task(self, coro: Awaitable[Any], user_id: str | None) -> None:
        while len(self._active) >= self.max_active:
            await asyncio.sleep(0.1)
        task = asyncio.create_task(coro)
        self._active.add(task)
        if user_id:
            self._user_tasks[user_id].add(task)
        task.add_done_callback(lambda t, u=user_id: self._finish_task(t, u))

    def _finish_task(self, task: asyncio.Task, user_id: str | None) -> None:
        self._active.discard(task)
        if user_id:
            self._user_tasks[user_id].discard(task)
        self._logger.info("Waiting tasks: %d", self._queue.qsize())

    def schedule(self, coro: Awaitable[Any], user_id: str | None = None) -> None:
        """Schedule a coroutine for execution."""
        self._queue.put_nowait((coro, user_id))
        self._logger.info("Waiting tasks: %d", self._queue.qsize())

    def cancel_user_tasks(self, user_id: str) -> None:
        """Cancel all tasks belonging to a specific user."""
        tasks = self._user_tasks.pop(user_id, set())
        for task in tasks:
            task.cancel()

    async def shutdown(self) -> None:
        """Cancel all running tasks and stop the scheduler."""
        while not self._queue.empty():
            await self._queue.get()
        for task in list(self._active):
            task.cancel()
        with suppress(Exception):
            await asyncio.gather(*self._active, return_exceptions=True)
        self._runner.cancel()
        with suppress(asyncio.CancelledError):
            await self._runner


scheduler = TaskScheduler()
