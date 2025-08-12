"""Entropy and perplexity helpers for GENESIS orchestrator."""

import math
import pickle
from collections import Counter
from pathlib import Path

from .orchestrator import dataset_dir as CONFIG_DATASET_DIR
from .genesis_trainer import GPT, GPTConfig


cached_model = None
cached_stoi = None


class MarkovEntropyCalculator:
    """Incrementally compute the Markov entropy of a text stream."""

    def __init__(self, n: int = 2):
        self.n = max(1, n)
        self.counts: Counter[str] = Counter()
        self._buffer = ""

    def update(self, chunk: str) -> None:
        if not chunk:
            return
        data = self._buffer + chunk
        if len(data) < self.n:
            self._buffer = data
            return
        for i in range(len(data) - self.n + 1):
            self.counts[data[i:i + self.n]] += 1
        self._buffer = data[-(self.n - 1):] if self.n > 1 else ""

    def entropy(self) -> float:
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log2(c / total) for c in self.counts.values())


class PerplexityCalculator:
    """Incrementally compute perplexity using the trained mini-GPT model."""

    def __init__(self):
        weights_path = Path(__file__).with_name("weights") / "model.pth"
        if not weights_path.exists():
            self.model = None
            return
        try:  # torch is optional
            import torch

            global cached_model, cached_stoi
            if cached_model is None or cached_stoi is None:
                checkpoint = torch.load(weights_path, map_location="cpu")
                model_args = checkpoint.get("model_args")
                if not model_args:
                    self.model = None
                    return
                model = GPT(GPTConfig(**model_args))
                model.load_state_dict(checkpoint["model"])
                model.eval()
                with open(Path(CONFIG_DATASET_DIR) / "meta.pkl", "rb") as f:
                    meta = pickle.load(f)
                cached_model = model
                cached_stoi = meta["stoi"]
            self.model = cached_model
            self.stoi = cached_stoi
        except Exception:  # pragma: no cover - torch not available
            self.model = None

        self.context: list[int] = []
        self.total_log_loss = 0.0
        self.total_tokens = 0

    def update(self, chunk: str) -> None:
        if not self.model or not chunk:
            return
        import torch

        for ch in chunk:
            token = self.stoi.get(ch, 0)
            if self.context:
                ids = torch.tensor([self.context[-1], token], dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    _, loss = self.model(ids[:, :-1], ids[:, 1:])
                self.total_log_loss += float(loss.item())
                self.total_tokens += 1
            self.context.append(token)
            if len(self.context) > 1:
                self.context = self.context[-1:]

    def perplexity(self) -> float:
        if not self.model or self.total_tokens == 0:
            return 0.0
        return float(math.exp(self.total_log_loss / self.total_tokens))


# Default set of file extensions considered for processing is kept in symphony


def markov_entropy(text: str, n: int = 2) -> float:
    calc = MarkovEntropyCalculator(n)
    calc.update(text)
    return calc.entropy()


def model_perplexity(text: str) -> float:
    """Return the perplexity of ``text`` under the trained mini-GPT model."""
    calc = PerplexityCalculator()
    calc.update(text)
    return calc.perplexity()
