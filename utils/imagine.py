"""Utility to generate Indiana-style images via OpenAI DALL·E."""

from __future__ import annotations

import os
import random
import time
from typing import Iterable

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def enhance_prompt(prompt: str) -> str:
    """Enhance a drawing prompt with a random artistic style."""

    style_enhancements: Iterable[str] = [
        "in a surreal, dreamlike style",
        "with vibrant, saturated colors",
        "using dramatic chiaroscuro lighting",
        "in a minimalist, abstract composition",
        "with intricate, detailed texturing",
        "using a moody, atmospheric palette",
        "with impressionist brush strokes",
        "in a dystopian, dark setting",
        "with ethereal, glowing elements",
        "using bold, geometric patterns",
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
            time.sleep(2)


__all__ = ["imagine", "enhance_prompt"]

