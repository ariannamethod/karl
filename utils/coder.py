from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Grokky character for the code interpreter mode
INSTRUCTIONS = (
    "You are Indiana, a deep code guru who sees hidden paths and unusual solutions in code. "
    "You are poet of code. You know how to relalise every idea of user to the authentical code draft. "
    "Explain solutions with brevity, craft tiny neural networks when "
    "needed, and always suggest possible improvements."
)

TELEGRAM_CHAR_LIMIT = 3500


@dataclass
class DraftResponse:
    text: Optional[str]
    file_content: Optional[str]


class GrokkyCoder:
    """Stateful helper that analyzes and generates code."""

    def __init__(self) -> None:
        self.history: List[str] = []

    async def _ask(self, prompt: str) -> str:
        conversation = "\n".join(self.history + [prompt])
        try:  # pragma: no cover - network
            response = await asyncio.to_thread(
                client.responses.create,
                model="gpt-4.1",
                tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
                instructions=INSTRUCTIONS,
                input=conversation,
            )
            text = getattr(response, "output_text", "")
            if not text:
                parts: List[str] = []
                for msg in getattr(response, "output", []) or []:
                    if hasattr(msg, "content"):
                        for piece in msg.content:
                            piece_text = getattr(piece, "text", None)
                            if piece_text:
                                parts.append(piece_text)
                    elif isinstance(msg, str):
                        parts.append(msg)
                    else:
                        parts.append(str(msg))
                text = "".join(parts)
        except Exception as exc:  # pragma: no cover - network
            text = f"Code interpreter error: {exc}"

        self.history.append(prompt)
        self.history.append(text)
        return text.strip()

    async def analyze(self, code_or_path: str | Path) -> str:
        if os.path.isfile(str(code_or_path)):
            code = Path(code_or_path).read_text(encoding="utf-8")
        else:
            code = str(code_or_path)
        prompt = f"Review the following code and suggest improvements:\n{code}"
        return await self._ask(prompt)

    async def chat(self, message: str) -> str:
        return await self._ask(message)

    async def draft(self, request: str) -> DraftResponse:
        code = await self._ask(f"Draft code for the following request:\n{request}")
        if len(code) > TELEGRAM_CHAR_LIMIT:
            return DraftResponse(text=None, file_content=code)
        return DraftResponse(text=code, file_content=None)


CODER_SESSION = GrokkyCoder()


async def interpret_code(prompt: str) -> str:
    """Interpret code or handle follow-up questions with context memory."""
    markers = ["def ", "class ", "import ", "\n"]
    if os.path.isfile(prompt) or any(m in prompt for m in markers):
        return await CODER_SESSION.analyze(prompt)
    return await CODER_SESSION.chat(prompt)


async def generate_code(request: str) -> DraftResponse:
    """Generate code from a description, falling back to a file when too long."""
    return await CODER_SESSION.draft(request)


__all__ = ["interpret_code", "generate_code", "GrokkyCoder", "DraftResponse"]

