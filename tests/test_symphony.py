import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from GENESIS_orchestrator.symphony import collect_new_data, markov_entropy  # noqa: E402


def test_binary_files_are_skipped(tmp_path):
    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"\x00\x01\x02")
    ready, data = collect_new_data([tmp_path], tmp_path / "out.txt", threshold=1)
    assert ready is False
    assert data == ""


def test_markov_entropy_on_simple_strings():
    assert markov_entropy("aaaa", n=1) == 0.0
    assert round(markov_entropy("abcabc", n=1), 2) == 1.58


def test_markov_entropy_with_unicode_symbols():
    text = "😀😃😀😃"
    # Two unique symbols appear twice -> probabilities 0.5 each -> entropy 1 bit
    assert markov_entropy(text, n=1) == 1.0


def test_markov_entropy_when_n_exceeds_length():
    text = "ab"
    # n is capped to len(text)=2 giving a single 2-gram -> entropy 0
    assert markov_entropy(text, n=5) == 0.0


def test_collect_new_data_with_threshold(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("hi")
    dataset = tmp_path / "out.txt"
    ready, data = collect_new_data([tmp_path], dataset, threshold=1)
    assert ready is True
    assert data == "hi"
    assert dataset.read_text() == "hi"


def test_collect_new_data_without_threshold(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("hi")
    dataset = tmp_path / "out.txt"
    ready, data = collect_new_data([tmp_path], dataset, threshold=10)
    assert ready is False
    assert data == "hi"
    assert not dataset.exists()
