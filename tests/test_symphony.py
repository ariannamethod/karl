import pytest

from GENESIS_orchestrator.entropy import markov_entropy, ngram_counter
from GENESIS_orchestrator.symphony import collect_new_data

def test_markov_entropy_simple():
    text = "abab"
    assert round(markov_entropy(text, n=2), 2) == 0.92


def test_markov_entropy_empty_and_short():
    assert markov_entropy("", n=3) == 0.0
    assert markov_entropy("a", n=3) == 0.0


def test_markov_entropy_non_string():
    with pytest.raises(TypeError):
        markov_entropy(123)  # type: ignore[arg-type]


def test_ngram_counter_stream():
    stream = ["aba", "ba"]
    gen = ngram_counter(stream, n=2)
    first = next(gen)
    assert first == {"ab": 1, "ba": 1}
    second = next(gen)
    assert second == {"ab": 2, "ba": 2}


def test_markov_entropy_stream():
    stream = ["ab", "ab"]
    assert round(markov_entropy(stream, n=2), 2) == 0.92

def test_collect_new_data(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("hello world")
    ready, data = collect_new_data([tmp_path], tmp_path / "out.txt", threshold=5)
    assert ready is True
    assert "hello" in data
    assert (tmp_path / "out.txt").exists()


def test_collect_new_data_any_extension(tmp_path):
    file = tmp_path / "script.py"
    file.write_text("print('ok')")
    ready, data = collect_new_data([tmp_path], tmp_path / "out.txt", threshold=1)
    assert ready is True
    assert "print" in data


def test_collect_new_data_excludes_dataset(tmp_path):
    dataset = tmp_path / "out.txt"
    dataset.write_text("old")
    file = tmp_path / "new.txt"
    file.write_text("new text")
    ready, data = collect_new_data([tmp_path], dataset, threshold=1)
    assert ready is True
    assert "old" not in data
