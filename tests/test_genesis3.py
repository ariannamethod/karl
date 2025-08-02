import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import genesis3  # noqa: E402


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, *args, **kwargs):
        data = {"choices": [{"message": {"content": "deep insight"}}]}
        return DummyResponse(data)


@pytest.mark.asyncio
async def test_genesis3_deep_dive(monkeypatch):
    monkeypatch.setenv("PPLX_API_KEY", "TOKEN")
    monkeypatch.setattr(genesis3, "httpx", type("x", (), {"AsyncClient": DummyClient}))
    result = await genesis3.genesis3_deep_dive("thought", "prompt")
    assert result == "🔍 deep insight"
