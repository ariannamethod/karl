import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.memory import MemoryManager  # noqa: E402
from utils.vectorstore import LocalVectorStore  # noqa: E402


@pytest.mark.asyncio
async def test_memory_save_and_retrieve(tmp_path):
    db_path = tmp_path / "memory.db"
    vec_path = tmp_path / "vector.json"
    store = LocalVectorStore(persist_path=str(vec_path))
    memory = MemoryManager(db_path=str(db_path), vectorstore=store)

    await memory.save("u1", "q1", "r1")
    if memory._tasks:
        await asyncio.gather(*memory._tasks)

    retrieved = await memory.retrieve("u1", "q1")
    assert "r1" in retrieved

    results = await memory.search_memory("u1", "r1")
    assert any("r1" in r for r in results)

    await memory.close()


@pytest.mark.asyncio
async def test_prune_user_records(tmp_path):
    db_path = tmp_path / "memory.db"
    store = LocalVectorStore()
    memory = MemoryManager(db_path=str(db_path), vectorstore=store, max_records_per_user=2)

    for i in range(3):
        await memory.save("u1", f"q{i}", f"r{i}")
    if memory._tasks:
        await asyncio.gather(*memory._tasks)

    db = await memory.connect()
    async with db.execute("SELECT COUNT(*) FROM memory WHERE user_id=?", ("u1",)) as cur:
        count = (await cur.fetchone())[0]
    assert count == 2

    responses = await memory.retrieve("u1", "q")
    assert "r0" not in responses
    assert "r1" in responses and "r2" in responses

    await memory.close()
