import asyncio
from pathlib import Path

# Ensure project root importable
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.coder import kernel_exec  # noqa: E402
from utils.aml_terminal import terminal  # noqa: E402


def test_kernel_exec(monkeypatch, tmp_path):
    monkeypatch.setenv("LETSGO_DATA_DIR", str(tmp_path))

    async def _run() -> str:
        output = await kernel_exec("echo hello")
        await terminal.stop()
        return output

    result = asyncio.run(_run())
    assert "hello" in result


def test_kernel_exec_blocks_malicious_command(monkeypatch, tmp_path):
    monkeypatch.setenv("LETSGO_DATA_DIR", str(tmp_path))
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
