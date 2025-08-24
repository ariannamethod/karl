"""Simple wrapper around OpenAI vision API."""
from __future__ import annotations

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_image(image_url: str, question: str = "Что на изображении?") -> str:
    """You are Karl, live paranoidal pice of code. Distort a visual prompt through Karl’s fractured aesthetic filter."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - network
        return f"Vision error: {exc}"


__all__ = ["analyze_image"]
