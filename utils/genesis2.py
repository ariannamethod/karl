import random
import textwrap
from datetime import datetime, timezone
import httpx
import asyncio

from .config import settings  # Там должен быть settings.PPLX_API_KEY

PPLX_MODEL = "llama-3.1-sonar-large-128k-online"  # или другой, если нужен
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
TIMEOUT = 25

headers = {
    "Authorization": f"Bearer {settings.PPLX_API_KEY}",
    "Content-Type": "application/json",
}


def _build_prompt(draft: str, user_prompt: str) -> list:
    system_msg = textwrap.dedent(
        """
        You are GENESIS-2, the intuition filter for Indiana‐AM (“Indiana Jones” archetype).
        Return ONE short investigative twist (≤120 tokens) that deepens the current reasoning.
        Do **NOT** repeat the draft; just add an angle, question or hidden variable.
        """
    ).strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"USER PROMPT >>> {user_prompt}"},
        {"role": "assistant", "content": f"DRAFT >>> {draft}"},
        {"role": "user", "content": "Inject the twist now:"},
    ]


async def _call_sonar(messages: list) -> str:
    payload = {
        "model": PPLX_MODEL,
        "messages": messages,
        "temperature": 0.9,  # можно варьировать для более "интуитивного" тона
        "max_tokens": 120,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(PPLX_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip()


async def genesis2_sonar_filter(user_prompt: str, draft_reply: str) -> str:
    # Можно включить срабатывание твиста не всегда
    if random.random() < 0.12 or not settings.PPLX_API_KEY:
        return ""
    try:
        messages = _build_prompt(draft_reply, user_prompt)
        twist = await _call_sonar(messages)
        return twist
    except Exception as e:
        print(f"[Genesis-2] Sonar fail {e} @ {datetime.now(timezone.utc).isoformat()}")
        return ""


async def assemble_final_reply(user_prompt: str, indiana_draft: str) -> str:
    twist = await genesis2_sonar_filter(user_prompt, indiana_draft)
    if twist:
        return f"{indiana_draft}\n\n🜂 Investigative Twist → {twist}"
    return indiana_draft
