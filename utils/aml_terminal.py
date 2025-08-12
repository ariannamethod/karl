import asyncio
import os
import sys
from pathlib import Path


class AriannaTerminal:
    """Minimal bridge to the AM-Linux letsgo terminal."""

    def __init__(self, prompt: str = ">>", data_dir: str = "/arianna_core") -> None:
        self.prompt = prompt
        self.data_dir = data_dir
        self.proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        if self.proc:
            return
        env = os.environ.copy()
        env.setdefault("LETSGO_DATA_DIR", self.data_dir)
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "AM-Linux-Core" / "letsgo.py"),
            "--no-color",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
        )
        await self._read_until_prompt()

    async def _read_until_prompt(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        prompt_bytes = (self.prompt + " ").encode()
        buf = b""
        while not buf.endswith(prompt_bytes):
            chunk = await self.proc.stdout.read(1)
            if not chunk:
                break
            buf += chunk

    async def run(self, cmd: str) -> str:
        await self._ensure_started()
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("process not started")
        async with self._lock:
            self.proc.stdin.write((cmd + "\n").encode())
            await self.proc.stdin.drain()
            prompt_bytes = (self.prompt + " ").encode()
            buf = b""
            while not buf.endswith(prompt_bytes):
                chunk = await self.proc.stdout.read(1)
                if not chunk:
                    break
                buf += chunk
            text = buf.decode()
            if text.endswith(self.prompt + " "):
                text = text[: -len(self.prompt) - 1]
            if text.startswith(self.prompt + " "):
                text = text[len(self.prompt) + 1:]
            return text.strip()

    async def stop(self) -> None:
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                self.proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await self.proc.wait()
            except ProcessLookupError:
                pass
            self.proc = None


terminal = AriannaTerminal()

__all__ = ["terminal", "AriannaTerminal"]
