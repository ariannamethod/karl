import sys
from pathlib import Path
import random
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import genesis2  # noqa: E402


def test_build_prompt():
    draft = "draft answer"
    user = "question?"
    messages = genesis2._build_prompt(draft, user, "en")
    assert messages[0]["role"] == "system"
    assert "GENESIS-2" in messages[0]["content"]
    assert messages[1]["content"].endswith(user)
    assert messages[2]["content"].endswith(draft)


@pytest.mark.asyncio
async def test_genesis2_sonar_filter(monkeypatch):
    monkeypatch.setattr(genesis2.settings, "PPLX_API_KEY", "TOKEN")
    monkeypatch.setattr(random, "random", lambda: 0.5)

    async def fake_call(messages):
        return "twist!"

    monkeypatch.setattr(genesis2, "_call_sonar", fake_call)

    result = await genesis2.genesis2_sonar_filter("user", "draft", "en")
    assert result == "twist!"


@pytest.mark.asyncio
async def test_genesis2_sonar_filter_disabled(monkeypatch):
    monkeypatch.setattr(genesis2.settings, "PPLX_API_KEY", "")
    monkeypatch.setattr(random, "random", lambda: 0.5)

    result = await genesis2.genesis2_sonar_filter("user", "draft", "en")
    assert result == ""


@pytest.mark.asyncio
async def test_assemble_final_reply(monkeypatch):
    async def fake_filter(user_prompt, draft_reply, language):
        return "twist!"

    monkeypatch.setattr(genesis2, "genesis2_sonar_filter", fake_filter)
    result = await genesis2.assemble_final_reply("q", "ans", "en")
    assert "twist!" in result
