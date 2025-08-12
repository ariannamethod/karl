import asyncio
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.coder import (  # noqa: E402
    ACCESS_DENIED_MESSAGE,
    IndianaCoder,
    format_core_commands,
    CORE_COMMANDS,
    generate_code,
    kernel_exec,
)


def test_analyze_denies_outside_repo(tmp_path):
    outside_file = tmp_path / "code.py"
    outside_file.write_text("print('hi')", encoding="utf-8")

    coder = IndianaCoder()
    result = asyncio.run(coder.analyze(str(outside_file)))
    assert result == ACCESS_DENIED_MESSAGE


def test_analyze_allows_repo_file(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    inside_file = repo_root / "tmp_test_file.py"
    inside_file.write_text("print('hi')", encoding="utf-8")

    coder = IndianaCoder()

    async def fake_ask(self, prompt: str) -> str:  # pragma: no cover - simple stub
        return "OK"

    monkeypatch.setattr(IndianaCoder, "_ask", fake_ask)

    try:
        result = asyncio.run(coder.analyze(str(inside_file)))
        assert result == "OK"
    finally:
        inside_file.unlink()


def test_help_lists_core_commands():
    """Ensure /help returns entries from CORE_COMMANDS."""
    help_text = format_core_commands()
    assert "/help" in help_text
    assert any(cmd in help_text for cmd in CORE_COMMANDS)


def test_generate_code_short(monkeypatch):
    async def fake_ask(self, prompt: str) -> str:
        return "print('hi')"

    monkeypatch.setattr(IndianaCoder, "_ask", fake_ask)
    result = asyncio.run(generate_code("say hi"))
    assert result.text == "print('hi')"
    assert result.file_content is None


def test_generate_code_long(monkeypatch):
    long_code = "x" * 4000

    async def fake_ask(self, prompt: str) -> str:
        return long_code

    monkeypatch.setattr(IndianaCoder, "_ask", fake_ask)
    result = asyncio.run(generate_code("long code"))
    assert result.text is None
    assert result.file_content == long_code


def test_kernel_exec_blocks(monkeypatch):
    logs = {}

    class FakeTerminal:
        def __init__(self):
            self.ran = False
            self.stopped = False

        async def run(self, cmd: str):
            self.ran = True
            return "ran"  # pragma: no cover - not expected

        async def stop(self):
            self.stopped = True

    async def fake_filter(command: str, base: str, lang: str) -> str:
        return "twist"

    fake_term = FakeTerminal()
    monkeypatch.setattr("utils.coder.terminal", fake_term)
    monkeypatch.setattr("utils.coder.is_blocked", lambda cmd: True)
    monkeypatch.setattr("utils.coder.log_blocked", lambda cmd: logs.setdefault("cmd", cmd))
    monkeypatch.setattr("utils.coder.genesis2_sonar_filter", fake_filter)

    result = asyncio.run(kernel_exec("rm -rf /"))
    assert "Терминал закрыт" in result
    assert "twist" in result
    assert logs["cmd"] == "rm -rf /"
    assert fake_term.stopped is True
    assert fake_term.ran is False


def test_kernel_exec_runs(monkeypatch):
    class FakeTerminal:
        def __init__(self):
            self.cmd = None
            self.stopped = False

        async def run(self, cmd: str):
            self.cmd = cmd
            return "ok"

        async def stop(self):
            self.stopped = True

    async def raise_filter(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("genesis2_sonar_filter called")

    fake_term = FakeTerminal()
    monkeypatch.setattr("utils.coder.terminal", fake_term)
    monkeypatch.setattr("utils.coder.is_blocked", lambda cmd: False)
    monkeypatch.setattr("utils.coder.genesis2_sonar_filter", raise_filter)

    result = asyncio.run(kernel_exec("echo hi"))
    assert result == "ok"
    assert fake_term.cmd == "/run echo hi"
    assert fake_term.stopped is False
