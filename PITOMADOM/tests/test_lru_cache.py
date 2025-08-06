import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from server import LRUCacheTTL  # noqa: E402


def test_stale_entries_removed_after_ttl():
    cache = LRUCacheTTL(max_items=10, ttl=0.1)
    cache.set("a", 1)
    time.sleep(0.2)
    assert cache.get("a") is None
    assert len(cache) == 0


def test_oldest_keys_evicted_over_capacity():
    cache = LRUCacheTTL(max_items=2, ttl=100)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2
