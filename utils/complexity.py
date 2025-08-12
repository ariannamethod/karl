from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ThoughtComplexityLogger:
    """Log complexity scale and entropy for each turn."""

    def __init__(self):
        self.logs = []  # timestamp, message, scale, entropy

    def log_turn(self, message: str, complexity_scale: int, entropy: float) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": message,
            "complexity_scale": complexity_scale,
            "entropy": float(entropy),
        }
        self.logs.append(record)
        logger.info(
            f"LOG@{record['timestamp']} | Complexity: {complexity_scale} | Entropy: {entropy:.3f}"
        )

    def recent(self, n: int = 7):
        return self.logs[-n:]


def estimate_complexity_and_entropy(msg: str) -> tuple[int, float]:
    """Heuristic estimation of complexity (1-3) and entropy of a message."""
    complexity = 1
    lowered = msg.lower()
    if any(word in lowered for word in ["why", "paradox", "recursive", "self", "meta"]):
        complexity += 1
    if len(msg) > 300:
        complexity += 1
    complexity = min(3, complexity)
    entropy = min(1.0, float(len(set(msg.split()))) / 40.0)
    return complexity, entropy
