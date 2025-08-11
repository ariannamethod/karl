import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.vectorstore import LocalVectorStore  # noqa: E402


def test_local_vector_store_eviction():
    store = LocalVectorStore(max_size=2)

    async def _populate():
        await store.store("1", "text1")
        await store.store("2", "text2")
        await store.store("3", "text3")

    asyncio.run(_populate())

    # Oldest entry should be evicted
    assert "1" not in store._store
    assert list(store._store.keys()) == ["2", "3"]
