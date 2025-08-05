from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.complexity import estimate_complexity_and_entropy  # noqa: E402


def test_complexity_basic():
    c, e = estimate_complexity_and_entropy("hello world")
    assert c == 1
    assert 0 <= e <= 1


def test_complexity_deep():
    msg = "why " * 80 + "paradox"  # ensures length and keyword
    c, _ = estimate_complexity_and_entropy(msg)
    assert c == 3
