"""Utility to generate KARL-style images via OpenAI DALL·E."""

from __future__ import annotations

import os
import random
import time
from typing import Iterable

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def enhance_prompt(prompt: str) -> str:
    """You are Karl. Distort a visual prompt through Karl’s fractured aesthetic filter."""

    style_enhancements: Iterable[str] = [
        "as if glimpsed during a fever dream",
        "with unnerving color harmonies and subtle visual tension",
        "in an uncanny blend of sacred geometry and broken realism",
        "using chiaroscuro that hides more than it reveals",
        "like a false memory rendered in vivid detail",
        "with visual glitches and impossible perspectives",
        "as an echo of a painting that never existed",
        "in the palette of an abandoned cathedral's stained glass",
        "with dream logic and collapsing dimensions",
        "in a style reminiscent of a post-apocalyptic instruction manual",
    ]

    if len(prompt.split()) > 15:
        return prompt

    enhancement = random.choice(list(style_enhancements))
    return f"{prompt.rstrip('.!?')} — {enhancement}."


def imagine(prompt: str, size: str = "1024x1024") -> str:
    """Generate an image URL from a text prompt."""

    enhanced_prompt = enhance_prompt(prompt)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                n=1,
                size=size,
            )
            return response.data[0].url
        except Exception as exc:  # pragma: no cover - network
            if attempt == max_retries - 1:
                return f"Image generation error: {exc}"
            time.sleep(2 ** attempt)


__all__ = ["imagine", "enhance_prompt"]
