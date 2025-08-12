
import importlib.util
import sys
from pathlib import Path

utils_spec = importlib.util.spec_from_file_location(
    "_test_utils", Path(__file__).with_name("utils.py")
)
_test_utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = _test_utils
utils_spec.loader.exec_module(_test_utils)
_write_log = _test_utils._write_log

spec = importlib.util.spec_from_file_location(
    "letsgo", Path(__file__).resolve().parents[1] / "letsgo.py"
)
letsgo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = letsgo
spec.loader.exec_module(letsgo)


def test_summarize_large_log(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    # create large log file with many matching lines
    lines = [f"{i} match" for i in range(10000)]
    _write_log(log_dir, "big", lines)
    monkeypatch.setattr(letsgo, "LOG_DIR", log_dir)
    result = letsgo.summarize("match")
    expected = "\n".join(lines[-5:])
    assert result == expected
