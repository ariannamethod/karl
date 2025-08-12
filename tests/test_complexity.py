from utils.complexity import (
    ThoughtComplexityLogger,
    estimate_complexity_and_entropy,
)

def test_complexity_basic():
    c, e = estimate_complexity_and_entropy("hello world")
    assert c == 1
    assert 0 <= e <= 1

def test_complexity_deep():
    msg = "why " * 80 + "paradox"  # ensures length and keyword
    c, _ = estimate_complexity_and_entropy(msg)
    assert c == 3

def test_log_turn_records_message():
    logger = ThoughtComplexityLogger()
    logger.log_turn("hello", 2, 0.5)
    assert logger.logs[-1]["message"] == "hello"
