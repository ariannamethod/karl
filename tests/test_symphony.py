from pathlib import Path

from GENESIS_orchestrator.symphony import collect_new_data, markov_entropy

def test_markov_entropy_simple():
    text = "abab"
    assert round(markov_entropy(text, n=2), 2) == 0.92

def test_collect_new_data(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("hello world")
    ready, data = collect_new_data([tmp_path], tmp_path / "out.txt", threshold=5)
    assert ready is True
    assert "hello" in data
    assert (tmp_path / "out.txt").exists()
