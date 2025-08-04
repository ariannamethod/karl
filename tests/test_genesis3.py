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


captured = []


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, *args, **kwargs):
        captured.append(kwargs.get("json"))
        data = {
            "choices": [
                {"message": {"content": "<think>prep</think>deep insight"}}
            ]
        }
        return DummyResponse(data)


@pytest.mark.asyncio
async def test_genesis3_deep_dive(monkeypatch):
    monkeypatch.setenv("PPLX_API_KEY", "TOKEN")
    monkeypatch.setattr(genesis3, "httpx", type("x", (), {"AsyncClient": DummyClient}))
    result = await genesis3.genesis3_deep_dive("thought", "prompt")
    assert result == "🔍 deep insight..."
    assert "FOLLOWUP" not in captured[0]["messages"][1]["content"]

    await genesis3.genesis3_deep_dive("thought", "prompt", is_followup=True)
    assert "FOLLOWUP" in captured[1]["messages"][1]["content"]
