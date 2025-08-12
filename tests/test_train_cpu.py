import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from GENESIS_orchestrator import symphony  # noqa: E402


def test_train_model_uses_cpu(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, cwd=None, check=False, capture_output=False, text=False):
        captured['cmd'] = cmd
        captured['capture_output'] = capture_output
        captured['text'] = text

        class Result:
            stdout = ''
            stderr = ''
        return Result()

    monkeypatch.setattr(symphony.subprocess, 'run', fake_run)
    dataset = tmp_path / 'data'
    dataset.mkdir()
    symphony.train_model(dataset, tmp_path / 'out')
    cmd = captured['cmd']
    assert '--device=cpu' in cmd
    assert '--block_size=32' in cmd
    assert '--batch_size=4' in cmd
    assert '--n_layer=1' in cmd
    assert '--n_head=1' in cmd
    assert '--n_embd=32' in cmd
    assert captured['capture_output'] is True
    assert captured['text'] is True
