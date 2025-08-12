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
