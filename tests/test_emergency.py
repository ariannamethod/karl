from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


class DummyMessage:
    def __init__(self, text="/emergency"):
        self.text = text
        self.from_user = SimpleNamespace(id=123, language_code="en")
        self.chat = SimpleNamespace(id=123, type="private")
        self.voice = None
        self.photo: list = []
        self.answers: list[str] = []

    async def answer(self, text: str):
        self.answers.append(text)


@pytest.mark.asyncio
async def test_emergency_command_toggles_flag():
    main.EMERGENCY_MODE = False
    m = DummyMessage("/emergency")
    await main.toggle_emergency_mode(m)
    assert main.EMERGENCY_MODE is True
    assert any("emergency mode" in ans.lower() for ans in m.answers)


@pytest.mark.asyncio
async def test_emergency_command_deactivates():
    main.EMERGENCY_MODE = True
    m = DummyMessage("/emergency")
    await main.toggle_emergency_mode(m)
    assert main.EMERGENCY_MODE is False
    assert any("deactivated" in ans.lower() for ans in m.answers)


@pytest.mark.asyncio
async def test_emergency_routes_messages(monkeypatch):
    main.EMERGENCY_MODE = True

    run_calls = []

    async def fake_run(cmd: str) -> str:
        run_calls.append(cmd)
        return "terminal output"

    monkeypatch.setattr(main.terminal, "run", fake_run)

    m = DummyMessage("echo hi")
    await main.handle_message(m)

    assert run_calls == ["echo hi"]
    assert m.answers == ["terminal output"]
