import asyncio
import sys
from pathlib import Path


# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.coder import ACCESS_DENIED_MESSAGE, IndianaCoder  # noqa: E402


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
