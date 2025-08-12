"""Entropy and perplexity helpers for GENESIS orchestrator."""

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

from .orchestrator import dataset_dir as CONFIG_DATASET_DIR
from .genesis_trainer import GPT, GPTConfig


cached_model = None
cached_stoi = None

# Default set of file extensions considered for processing is kept in symphony

def markov_entropy(text: str, n: int = 2) -> float:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return 0.0
    n = max(1, min(n, len(text)))
    counts = Counter(text[i: i + n] for i in range(len(text) - n + 1))
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def model_perplexity(text: str) -> float:
    """Return the perplexity of ``text`` under the trained mini-GPT model."""
    weights_path = Path(__file__).with_name("weights") / "model.pth"
    if not text or not weights_path.exists():
        return 0.0
    import torch
    global cached_model, cached_stoi
    if cached_model is None or cached_stoi is None:
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            model_args = checkpoint.get("model_args")
            if not model_args:
                return 0.0
            model = GPT(GPTConfig(**model_args))
            model.load_state_dict(checkpoint["model"])
            model.eval()
            with open(Path(CONFIG_DATASET_DIR) / "meta.pkl", "rb") as f:
                meta = pickle.load(f)
            cached_model = model
            cached_stoi = meta["stoi"]
        except Exception:
            return 0.0
    encoded = [cached_stoi.get(ch, 0) for ch in text]
    if len(encoded) < 2:
        return 0.0
    ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        _, loss = cached_model(ids[:, :-1], ids[:, 1:])
    return float(math.exp(loss.item()))
