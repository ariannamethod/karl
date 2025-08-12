import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from GENESIS_orchestrator.genesis_trainer import train_model  # noqa: E402


def test_train_model_uses_cpu(monkeypatch, tmp_path):
    captured = {}

    def fake_train_loop(args):
        captured['args'] = args

    from GENESIS_orchestrator import genesis_trainer

    monkeypatch.setattr(genesis_trainer, 'train_loop', fake_train_loop)
    dataset = tmp_path / 'data'
    dataset.mkdir()
    train_model(dataset, tmp_path / 'out')
    args = captured['args']
    assert args.device == 'cpu'
    assert args.block_size == 32
    assert args.batch_size == 4
    assert args.n_layer == 1
    assert args.n_head == 1
    assert args.n_embd == 32
