import math
import pickle
import types
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from GENESIS_orchestrator.entropy import markov_entropy, model_perplexity  # noqa: E402
from GENESIS_orchestrator.genesis_trainer import prepare_char_dataset  # noqa: E402


def test_markov_entropy_empty_string():
    assert markov_entropy("") == 0.0


def test_markov_entropy_various_n():
    text = "ab"
    assert markov_entropy(text, n=1) == pytest.approx(1.0)
    assert markov_entropy(text, n=2) == pytest.approx(0.0)
    assert markov_entropy(text, n=5) == pytest.approx(0.0)


def test_prepare_char_dataset_creates_files(tmp_path):
    text = "abcd"
    prepare_char_dataset(text, tmp_path)

    train_file = tmp_path / "train.bin"
    val_file = tmp_path / "val.bin"
    meta_file = tmp_path / "meta.pkl"

    assert train_file.exists()
    assert val_file.exists()
    assert meta_file.exists()

    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    assert meta["vocab_size"] == 4
    assert meta["stoi"]["a"] == 0


def test_prepare_char_dataset_empty_text_raises(tmp_path):
    with pytest.raises(ValueError):
        prepare_char_dataset("", tmp_path)


def test_model_perplexity_with_mocks(tmp_path, monkeypatch):
    import GENESIS_orchestrator.entropy as entropy

    monkeypatch.setattr(entropy, "CONFIG_DATASET_DIR", tmp_path)
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump({"stoi": {"a": 1, "b": 2}}, f)

    weights_dir = Path(entropy.__file__).with_name("weights")
    weights_dir.mkdir(exist_ok=True)
    weight_file = weights_dir / "model.pth"
    weight_file.touch()

    fake_torch = types.SimpleNamespace()

    def load(path, map_location=None):
        return {"model_args": {"vocab_size": 2}, "model": {}}

    class FakeTensor:
        def unsqueeze(self, dim):
            return self

        def __getitem__(self, item):
            return self

    def tensor(data, dtype=None):
        return FakeTensor()

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch.load = load
    fake_torch.tensor = tensor
    fake_torch.no_grad = no_grad
    fake_torch.long = int

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

        def __call__(self, *args, **kwargs):

            class Loss:
                def item(self):
                    return math.log(5)
            return None, Loss()

    monkeypatch.setattr(entropy, "GPT", lambda cfg: FakeModel())
    monkeypatch.setattr(entropy, "GPTConfig", lambda **kwargs: kwargs)

    try:
        assert model_perplexity("ab") == pytest.approx(5.0)
    finally:
        try:
            weight_file.unlink()
            weights_dir.rmdir()
        except OSError:
            pass
