from GENESIS_orchestrator.genesis_trainer import train_model

def test_train_model_uses_cpu(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, cwd=None, check=False):
        captured['cmd'] = cmd

    from GENESIS_orchestrator import genesis_trainer

    # Force fallback to subprocess by simulating missing torch
    monkeypatch.setattr(genesis_trainer, 'torch', None)
    monkeypatch.setattr(genesis_trainer.subprocess, 'run', fake_run)
    dataset = tmp_path / 'data'
    dataset.mkdir()
    train_model(dataset, tmp_path / 'out')
    cmd = captured['cmd']
    assert '--device=cpu' in cmd
    assert '--block_size=32' in cmd
    assert '--batch_size=4' in cmd
    assert '--n_layer=1' in cmd
    assert '--n_head=1' in cmd
    assert '--n_embd=32' in cmd
