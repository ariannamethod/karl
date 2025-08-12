import httpx
import random
import textwrap
from datetime import datetime, timezone
import re
import json
import logging

from .config import settings  # settings.PPLX_API_KEY должен быть определён

logger = logging.getLogger(__name__)

PPLX_MODEL = "sonar-pro"
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
TIMEOUT = 20

headers = {
    "Authorization": f"Bearer {settings.PPLX_API_KEY}",
    "Content-Type": "application/json",
}

def _build_prompt(user_message: str, meta: dict, language: str) -> list:
    """
    Формирует системный промпт для GENESIS-6.
    meta: {'total_msgs': int, 'avg_pause_sec': float, 'silence_count': int, 'top_topics': list[str]}
    """
    system_msg = textwrap.dedent(f"""
        You are GENESIS-6, the silent psychological & resonance profiler for Indiana-AM.
        Your task for EACH user message:

        1. Emotional tone — 1-2 descriptive words.
        2. Behavioural/psychological pattern — short phrase.
        3. Resonance (0–100) — intuitive 'tuning' with scene.
        4. Recommendation for Indiana — adjust style/emotion/pacing.

        Output ONLY valid JSON:
        {{
          "emotional_tone": "...",
          "psych_pattern": "...",
          "resonance": <int>,
          "recommendation": "..."
        }}

        Reply in {language}.
    """).strip()

    meta_str = "\n".join([f"{k}: {v}" for k, v in meta.items()])
    combined_user = f"USER MESSAGE >>> {user_message}\nCONTEXT META >>>\n{meta_str}" if meta_str else f"USER MESSAGE >>> {user_message}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": combined_user},
    ]


async def _call_sonar(messages: list) -> dict:
    """Вызов Sonar для оценки и возврат профиля как dict"""
    payload = {
        "model": PPLX_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 200,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(PPLX_API_URL, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except Exception:
            logger.error("[Genesis-6] Sonar HTTP error: %s", resp.text)
            raise
        data = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(data)
        except Exception:
            return {"raw_response": data}


async def genesis6_profile_filter(user_message: str, meta: dict, language: str) -> dict:
    """
    Главная точка входа.
    user_message — текст пользователя
    meta — словарь с метриками (можно передавать пустой {} если нечего)
    """
    if not settings.PPLX_API_KEY:
        return {}
    try:
        messages = _build_prompt(user_message, meta, language)
        profile = await _call_sonar(messages)
        profile["timestamp"] = datetime.now(timezone.utc).isoformat()
        return profile
    except Exception as e:
        logger.error(
            f"[Genesis-6] fail {e} @ {datetime.now(timezone.utc).isoformat()}"
        )
        return {}
