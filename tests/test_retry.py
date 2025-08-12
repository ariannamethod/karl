import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from utils import genesis3  # noqa: E402


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


attempts = {"count": 0}


class FlakyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, *args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.HTTPError("boom")
        return DummyResponse({"choices": [{"message": {"content": "<think>x</think>done."}}]})


@pytest.mark.asyncio
async def test_genesis3_retry(monkeypatch):
    attempts["count"] = 0
    monkeypatch.setenv("PPLX_API_KEY", "TOKEN")
    monkeypatch.setattr(genesis3.httpx, "AsyncClient", FlakyClient)

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(genesis3.asyncio, "sleep", fake_sleep)

    result = await genesis3.genesis3_deep_dive("thought", "prompt")
    assert result == "🔍 done."
    assert attempts["count"] == 2
