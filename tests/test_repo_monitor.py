import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.repo_monitor as repo_monitor  # noqa: E402
from utils.repo_monitor import RepoWatcher  # noqa: E402


def test_on_change_error_logs(monkeypatch, tmp_path):
    (tmp_path / "file.txt").write_text("data", encoding="utf-8")

    def failing_callback():
        raise RuntimeError("boom")

    watcher = RepoWatcher([tmp_path], failing_callback)

    mock_logger = MagicMock()
    monkeypatch.setattr(repo_monitor, "logger", mock_logger)

    watcher.check_now()

    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert "on_change callback" in args[0]
    assert kwargs.get("exc_info") is True


def test_on_change_called_once(tmp_path):
    """Callback is triggered exactly once when a file changes."""
    file = tmp_path / "file.txt"
    file.write_text("first", encoding="utf-8")

    callback = MagicMock()
    watcher = RepoWatcher([tmp_path], callback)

    watcher._file_sha = watcher._scan()

    file.write_text("second", encoding="utf-8")
    watcher.check_now()

    callback.assert_called_once()
