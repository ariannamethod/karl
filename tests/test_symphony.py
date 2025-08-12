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


def test_main_reads_config_and_cli_override(monkeypatch, tmp_path):
    from GENESIS_orchestrator import symphony

    # Patch configuration values
    monkeypatch.setattr(symphony, "DEFAULT_THRESHOLD", 123)
    monkeypatch.setattr(symphony, "CONFIG_DATASET_DIR", tmp_path / "data")
    monkeypatch.setattr(symphony, "model_hyperparams", {})

    captured = {}

    def fake_collect(base_paths, dataset_path, threshold, resume):
        captured["threshold"] = threshold
        return True, "abc"

    def fake_prepare(text, dest):
        captured["dataset_dir"] = dest

    def fake_train(data_dir, out_dir):
        captured["train_dataset"] = data_dir

    monkeypatch.setattr(symphony, "collect_new_data", fake_collect)
    monkeypatch.setattr(symphony, "_prepare_char_dataset", fake_prepare)
    monkeypatch.setattr(symphony, "train_model", fake_train)

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))

    monkeypatch.setattr(sys, "argv", ["symphony.py"])
    symphony.main()
    assert captured["threshold"] == 123
    assert captured["dataset_dir"] == tmp_path / "data"
    assert captured["train_dataset"] == tmp_path / "data"

    monkeypatch.setattr(sys, "argv", ["symphony.py", "--threshold", "5", "--dataset_dir", str(tmp_path / "other")])
    symphony.main()
    assert captured["threshold"] == 5
    assert captured["dataset_dir"] == tmp_path / "other"
    assert captured["train_dataset"] == tmp_path / "other"
