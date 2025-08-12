import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.vectorstore import LocalVectorStore  # noqa: E402
from utils.config import settings  # noqa: E402


def test_local_vector_store_eviction():
    settings.OPENAI_API_KEY = ""
    store = LocalVectorStore(max_size=2)

    async def _populate():
        await store.store("1", "text1")
        await store.store("2", "text2")
        await store.store("3", "text3")

    asyncio.run(_populate())

    # Oldest entry should be evicted
    assert "1" not in store._store
    assert list(store._store.keys()) == ["2", "3"]


def test_local_vector_store_caching(monkeypatch):
    settings.OPENAI_API_KEY = ""
    store = LocalVectorStore()
    calls = {"count": 0}

    async def fake_generate(text: str):
        calls["count"] += 1
        return [ord(c) for c in text]

    monkeypatch.setattr(store, "_generate_embedding", fake_generate)

    async def _run():
        await store.store("1", "hello")
        await store.store("2", "hello")
        await store.search("world")
        await store.search("world")

    asyncio.run(_run())

    # one for storing "hello", one for first "world" query
    assert calls["count"] == 2


def test_search_max_docs_performance():
    settings.OPENAI_API_KEY = ""
    store = LocalVectorStore()

    async def _populate():
        for i in range(5000):
            await store.store(str(i), f"text{i}")
        # warm-up to cache query embedding
        await store.search("query", top_k=1, max_docs=1)

    asyncio.run(_populate())

    import time

    start = time.perf_counter()
    asyncio.run(store.search("query", top_k=5))
    full_time = time.perf_counter() - start

    start = time.perf_counter()
    asyncio.run(store.search("query", top_k=5, max_docs=100))
    limited_time = time.perf_counter() - start

    assert limited_time < full_time
