import asyncio
import pytest

from utils.context_neural_processor import parse_and_store_file

@pytest.mark.asyncio
async def test_parse_and_store_text(tmp_path):
    file = tmp_path / "example.txt"
    file.write_text("Hello Indiana")
    result = await parse_and_store_file(str(file))
    assert "Tags:" in result
    assert "Summary:" in result
