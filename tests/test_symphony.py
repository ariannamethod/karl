import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from GENESIS_orchestrator import orchestrator as state  # noqa: E402
from GENESIS_orchestrator.symphony import collect_new_data  # noqa: E402
from GENESIS_orchestrator.entropy import markov_entropy  # noqa: E402


def test_binary_files_are_skipped(tmp_path):
    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"\x00\x01\x02")
    ready, data = collect_new_data([tmp_path], tmp_path / "out.txt", threshold=1)
    assert ready is False
    assert data == ""


def test_collect_new_data_mixed_binary_and_text(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_FILE", tmp_path / "state.json")
    text_file = tmp_path / "a.txt"
    text_file.write_text("hello")
    binary = tmp_path / "b.bin"
    binary.write_bytes(b"\x00\x01\x02")
    dataset = tmp_path / "out.txt"
    ready, data = collect_new_data([tmp_path], dataset, threshold=1)
    assert ready is True
    assert data == "hello"
    assert dataset.read_text() == "hello"
    saved = json.loads((tmp_path / "state.json").read_text())
    files = saved["files"]
    assert str(text_file.resolve()) in files
    assert str(binary.resolve()) not in files


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


def test_run_orchestrator_returns_metrics(monkeypatch, tmp_path):
    from GENESIS_orchestrator import symphony

    monkeypatch.setattr(
        symphony, "collect_new_data", lambda *a, **k: (False, "abc")
    )
    monkeypatch.setattr(symphony, "markov_entropy", lambda text: 1.0)
    monkeypatch.setattr(symphony, "model_perplexity", lambda text: 2.0)
    monkeypatch.setattr(symphony, "prepare_char_dataset", lambda *a, **k: None)
    monkeypatch.setattr(symphony, "train_model", lambda *a, **k: None)

    metrics = symphony.run_orchestrator(dataset_dir=tmp_path)
    assert metrics == {"markov_entropy": 1.0, "model_perplexity": 2.0}


def test_main_reads_config_and_cli_override(monkeypatch, tmp_path):
    from GENESIS_orchestrator import symphony

    # Patch configuration values
    monkeypatch.setattr(symphony, "DEFAULT_THRESHOLD", 123)
    monkeypatch.setattr(symphony, "CONFIG_DATASET_DIR", tmp_path / "data")
    monkeypatch.setattr(symphony, "model_hyperparams", {})

    captured = {}

    def fake_collect(base_paths, dataset_path, threshold, resume, *, allow_ext, deny_ext):
        captured["threshold"] = threshold
        captured["allow_ext"] = list(allow_ext) if allow_ext else None
        captured["deny_ext"] = list(deny_ext) if deny_ext else None
        return True, "abc"

    def fake_prepare(text, dest):
        captured["dataset_dir"] = dest

    def fake_train(data_dir, out_dir):
        captured["train_dataset"] = data_dir

    monkeypatch.setattr(symphony, "collect_new_data", fake_collect)
    monkeypatch.setattr(symphony, "prepare_char_dataset", fake_prepare)
    monkeypatch.setattr(symphony, "train_model", fake_train)

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))

    monkeypatch.setattr(sys, "argv", ["symphony.py"])
    symphony.main()
    assert captured["threshold"] == 123
    assert captured["dataset_dir"] == tmp_path / "data"
    assert captured["train_dataset"] == tmp_path / "data"
    assert captured["allow_ext"] == list(symphony.DEFAULT_ALLOW_EXT)
    assert captured["deny_ext"] is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symphony.py",
            "--threshold",
            "5",
            "--dataset_dir",
            str(tmp_path / "other"),
            "--allow-ext",
            ".py",
            "--allow-ext",
            ".md",
            "--deny-ext",
            ".log",
        ],
    )
    symphony.main()
    assert captured["threshold"] == 5
    assert captured["dataset_dir"] == tmp_path / "other"
    assert captured["train_dataset"] == tmp_path / "other"
    assert captured["allow_ext"] == [".py", ".md"]
    assert captured["deny_ext"] == [".log"]


def test_prepare_char_dataset_empty_text_raises(tmp_path):
    from GENESIS_orchestrator.genesis_trainer import prepare_char_dataset

    with pytest.raises(ValueError, match="non-empty"):
        prepare_char_dataset("", tmp_path)
