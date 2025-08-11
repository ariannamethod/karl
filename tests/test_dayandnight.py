import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.vectorstore import LocalVectorStore  # noqa: E402
from utils import dayandnight  # noqa: E402


@pytest.mark.asyncio
async def test_store_and_fetch_last_day(monkeypatch):
    store = LocalVectorStore()
    monkeypatch.setattr(dayandnight, "vector_store", store)
    await dayandnight._store_last_day("2024-01-01", "hi")
    last = await dayandnight._fetch_last_day()
    assert last == "2024-01-01"
