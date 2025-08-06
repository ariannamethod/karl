import sys
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from server import BioOrchestraMemory  # noqa: E402


def test_memory_store(tmp_path):
    store = tmp_path / "mem.json"
    m = BioOrchestraMemory(path=str(store))
    m.remember("HeLLo", "WoRLD")
    m.learn("FeedBack")
    results = m.recall("hello")
    assert results[0]["response"] == "world"
    assert any("feedback" in r for r in m.recall("FEEDBACK"))


def test_memory_fuzzy_match(tmp_path):
    store = tmp_path / "mem.json"
    m = BioOrchestraMemory(path=str(store))
    m.remember("hello world", "response")
    results = m.recall("helo wrld", fuzzy=True)
    assert results and results[0]["prompt"] == "hello world"


def test_memory_concurrent_writes(tmp_path):
    store = tmp_path / "mem.json"
    m = BioOrchestraMemory(path=str(store))

    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            m.remember(f"p{i}", f"r{i}")
        except Exception as e:  # pragma: no cover - we want to record unexpected failures
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(m.records) == 10
    for i in range(10):
        assert {"prompt": f"p{i}", "response": f"r{i}"} in m.records
