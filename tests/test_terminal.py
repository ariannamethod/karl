import asyncio
from pathlib import Path

# Ensure project root importable
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.coder import kernel_exec  # noqa: E402
from utils.aml_terminal import terminal  # noqa: E402


def test_kernel_exec_allowlist_hit(monkeypatch, tmp_path):
    monkeypatch.setenv("LETSGO_DATA_DIR", str(tmp_path))
    allow_file = Path("artefacts/allowed_commands.json")
    allow_file.write_text('["echo hello"]', encoding="utf-8")

    async def fake_run(cmd: str) -> str:
        assert cmd == "/run echo hello"
        return "hello"

    monkeypatch.setattr(terminal, "run", fake_run)

    async def _run() -> str:
        return await kernel_exec("echo hello")

    result = asyncio.run(_run())
    assert result == "hello"


def test_kernel_exec_blocks_malicious_command(monkeypatch, tmp_path):
    monkeypatch.setenv("LETSGO_DATA_DIR", str(tmp_path))
    allow_file = Path("artefacts/allowed_commands.json")
    allow_file.write_text('["rm -rf /"]', encoding="utf-8")
    log_file = Path("artefacts/blocked_commands.log")
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")

    async def fake_run(cmd: str) -> str:
        raise AssertionError("run should not be called")

    monkeypatch.setattr(terminal, "run", fake_run)

    async def _run() -> str:
        return await kernel_exec("rm -rf /")

    result = asyncio.run(_run())
    assert "Терминал закрыт" in result
    assert log_file.exists()
    assert "rm -rf /" in log_file.read_text(encoding="utf-8")


def test_kernel_exec_blocks_disallowed_command(monkeypatch, tmp_path):
    monkeypatch.setenv("LETSGO_DATA_DIR", str(tmp_path))
    allow_file = Path("artefacts/allowed_commands.json")
    allow_file.write_text('["echo hi"]', encoding="utf-8")
    log_file = Path("artefacts/blocked_commands.log")
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")

    async def fake_run(cmd: str) -> str:
        raise AssertionError("run should not be called")

    monkeypatch.setattr(terminal, "run", fake_run)

    async def _run() -> str:
        return await kernel_exec("echo hello")

    result = asyncio.run(_run())
    assert "Терминал закрыт" in result
    assert log_file.exists()
    assert "echo hello" in log_file.read_text(encoding="utf-8")
