import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from server import LRUCacheTTL, RateLimiter  # noqa: E402


class DummyEvict:
    def __init__(self):
        self.keys: list[str] = []

    def __call__(self, key: str) -> None:  # pragma: no cover - simple callback
        self.keys.append(key)


def test_lru_cache_add_and_get():
    cache = LRUCacheTTL(max_items=10, ttl=5)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert len(cache) == 1


def test_lru_cache_ttl_expiry_and_callback():
    evicted = DummyEvict()
    cache = LRUCacheTTL(max_items=10, ttl=1, evict_cb=evicted)
    cache.set("a", 1)
    time.sleep(1.2)
    # Access to trigger purge
    assert cache.get("a") is None
    assert len(cache) == 0
    assert evicted.keys == ["a"]


def test_lru_cache_respects_max_items():
    evicted = DummyEvict()
    cache = LRUCacheTTL(max_items=2, ttl=5, evict_cb=evicted)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 2
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert evicted.keys == ["a"]


def test_rate_limiter_exhaust_and_recover():
    limiter = RateLimiter(capacity=2, refill_per_sec=1)
    key = "client"
    assert limiter.allow(key)
    assert limiter.allow(key)
    assert not limiter.allow(key)
    time.sleep(1.2)
    assert limiter.allow(key)
