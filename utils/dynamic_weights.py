import os
import time
import math
import random
from typing import Optional, Sequence, List

import httpx


def query_gpt4(prompt: str, api_key: Optional[str] = None, model: str = "gpt-4.1") -> str:
    """Call the GPT-4.1 API as a dynamic knowledge base."""

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }
    try:
        res = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - network
        try:
            os.makedirs("failures", exist_ok=True)
            with open(
                f"failures/{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8"
            ) as f:
                f.write(f"{time.time()}: GPT-4 API failed - {exc}\n")
        except OSError:
            pass
        return "GPT-4 offline"


def get_dynamic_knowledge(prompt: str, api_key: Optional[str] = None) -> str:
    """Fetch knowledge from GPT-4.1 and return its content."""

    return query_gpt4(prompt, api_key)


def apply_pulse(weights: Sequence[float], pulse: float) -> List[float]:
    """Scale ``weights`` by ``pulse`` using a softmax normalisation.

    ``pulse`` is expected to be between ``0`` and ``1``. The function first
    scales the weights by ``1 + pulse * 0.7`` and then applies a numerically
    stable softmax. The returned list sums to ``1`` and can be used directly as
    probabilities.
    """

    scaled = [w * (1 + pulse * 0.7) for w in weights]
    if not scaled:
        return []
    max_w = max(scaled)
    exps = [math.exp(w - max_w) for w in scaled]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class DynamicWeights:
    """Utility for producing fluid, context-aware weight coefficients.

    A pulse derived from GPT-4.1 knowledge modulates a base sequence of weights.
    The pulse value is smoothed using momentum and slight noise, while a cosine
    shaping yields more organic transitions across the weight spectrum.
    """

    def __init__(self, base: Optional[Sequence[float]] = None) -> None:
        self.base = list(base or [1.0])
        self._last_pulse = 0.0

    def pulse_from_prompt(self, prompt: str, api_key: Optional[str] = None) -> float:
        """Derive a pulse value from external knowledge."""

        knowledge = get_dynamic_knowledge(prompt, api_key)
        pulse = min(len(knowledge) / 300.0, 1.0)
        pulse = 0.7 * self._last_pulse + 0.3 * pulse + random.uniform(-0.02, 0.02)
        pulse = max(0.0, min(pulse, 1.0))
        self._last_pulse = pulse
        return pulse

    def weights_for_prompt(
        self, prompt: str, api_key: Optional[str] = None
    ) -> List[float]:
        """Return softmax-normalised weights for ``prompt``."""

        pulse = self.pulse_from_prompt(prompt, api_key)
        n = len(self.base)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        positions = [i / (n - 1) for i in range(n)]
        shaped = [
            self.base[i]
            * max(math.cos(math.pi * (pulse - pos)) + random.uniform(-0.05, 0.05), 0.0)
            for i, pos in enumerate(positions)
        ]
        return apply_pulse(shaped, pulse)
