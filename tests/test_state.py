from GENESIS_orchestrator import orchestrator as state

def test_file_hash_consistency(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("content")
    h1 = state.file_hash(path)
    h2 = state.file_hash(path)
    assert h1 == h2
    path.write_text("different")
    h3 = state.file_hash(path)
    assert h3 != h1

def test_load_state_with_corrupted_json(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "state.json"
    bad.write_text("{not json")
    monkeypatch.setattr(state, "STATE_FILE", bad)
    with caplog.at_level("ERROR"):
        data = state.load_state()
    assert data == {}
    assert "failed to read state file" in caplog.text
