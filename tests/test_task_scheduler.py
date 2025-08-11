import asyncio as aio
import pytest

from utils.task_scheduler import TaskScheduler


@pytest.mark.asyncio
async def test_scheduler_limits_concurrency():
    scheduler = TaskScheduler(max_active=1)
    order = []

    async def job(i):
        await aio.sleep(0.05)
        order.append(i)

    scheduler.schedule(job(1))
    scheduler.schedule(job(2))

    await aio.sleep(0.06)
    assert order == [1]
    await aio.sleep(0.2)
    await scheduler.shutdown()
    assert order == [1, 2]


@pytest.mark.asyncio
async def test_cancel_user_tasks():
    scheduler = TaskScheduler(max_active=1)
    flag = {"ran": False}

    async def job():
        try:
            await aio.sleep(0.5)
            flag["ran"] = True
        except aio.CancelledError:
            raise

    scheduler.schedule(job(), user_id="u1")
    await aio.sleep(0.1)
    scheduler.cancel_user_tasks("u1")
    await aio.sleep(0.1)
    await scheduler.shutdown()
    assert not flag["ran"]
