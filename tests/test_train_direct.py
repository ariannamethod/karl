def test_train_model_runs_in_process(monkeypatch, tmp_path):
    from GENESIS_orchestrator import genesis_trainer

    called = {}

    def fake_run_training(dataset_dir, out_dir, device="cpu", hyperparams=None):
        called["dataset"] = dataset_dir
        called["out_dir"] = out_dir
        called["device"] = device
        called["hyperparams"] = hyperparams or {}

    monkeypatch.setattr(genesis_trainer, "run_training", fake_run_training)
    monkeypatch.setattr(genesis_trainer, "torch", object())

    dataset = tmp_path / "data"
    dataset.mkdir()

    genesis_trainer.train_model(dataset, tmp_path / "out")

    assert called["dataset"] == dataset
    assert called["out_dir"] == tmp_path / "out"
    assert called["device"] == "cpu"
    assert "block_size" in called["hyperparams"]
