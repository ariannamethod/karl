
import os
import re
import asyncio
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "letsgo", Path(__file__).resolve().parents[1] / "letsgo.py"
)
letsgo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = letsgo
spec.loader.exec_module(letsgo)

utils_spec = importlib.util.spec_from_file_location(
    "_test_utils", Path(__file__).with_name("utils.py")
)
_test_utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = _test_utils
utils_spec.loader.exec_module(_test_utils)
_write_log = _test_utils._write_log


def test_status_fields(monkeypatch):
    monkeypatch.setattr(letsgo, "_first_ip", lambda: "1.2.3.4")
    result = letsgo.status()
    lines = result.splitlines()
    assert len(lines) == 3
    expected_cpu = os.cpu_count()
    assert lines[0] == f"CPU cores: {expected_cpu}"
    assert re.match(r"^Uptime: \d+\.\d+s", lines[1])
    assert lines[2] == "IP: 1.2.3.4"


def test_summarize_no_logs(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    monkeypatch.setattr(letsgo, "LOG_DIR", log_dir)
    result = letsgo.summarize("anything")
    assert result == "no logs"


def test_summarize_term_filter(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write_log(log_dir, "sample", ["foo", "bar", "foo again", "baz"])
    monkeypatch.setattr(letsgo, "LOG_DIR", log_dir)
    result = letsgo.summarize("foo")
    assert result == "foo\nfoo again"
