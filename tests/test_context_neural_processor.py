import asyncio
from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.context_neural_processor import parse_and_store_file


@pytest.mark.asyncio
async def test_parse_and_store_text(tmp_path):
    file = tmp_path / "example.txt"
    file.write_text("Hello KARL")
    result = await parse_and_store_file(str(file))
    assert "Tags:" in result
    assert "Summary:" in result
