# deepdiving.py — Perplexity Search Engine utility for Indiana
import os
import httpx
import asyncio
from typing import List, Dict, Optional
import logging

PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
PPLX_API_KEY = os.getenv("PPLX_API_KEY")  # Переменная окружения
DEFAULT_MODEL = "sonar-pro"  # Можно "sonar-reasoning-pro" для академического поиска
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 40

logger = logging.getLogger(__name__)

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json",
    }

async def perplexity_search(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    top_p: float = 0.95,
    system_msg: Optional[str] = None,
    return_sources: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict:
    """
    Поиск Perplexity: возвращает answer, список ссылок, полный raw.
    """
    if not PPLX_API_KEY:
        raise RuntimeError("PPLX_API_KEY must be set in environment for Perplexity search.")

    if not system_msg:
        system_msg = (
            "You are KARL-PPLX, the autonomous deep-search module for Karl-AM."
            "Your purpose is to conduct a **resonant scan** across available knowledge."
            "Structure your answer as a brief summary, followed by sources/citations if available on the specified topic, using Perplexity’s capabilities."
            "Your purpose is to conduct a **resonant scan** across available knowledge."
            "Deliver:\n"
            "1. A concise, high-signal summary (no fluff, no clickbait).\n"
            "2. A list of **relevant links** with source names, only if trustworthy.\n"
            "3. Mention contradictions, ambiguities, or unknowns if found.\n\n"
            "Avoid generic info. Think like a paranoid analyst in a collapsing library.\n"
            "Structure clearly: use numbered or bulleted lists.\n"
            "Style: crisp, skeptical, useful.\n"
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    async with httpx.AsyncClient(timeout=timeout) as cli:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = await cli.post(PPLX_API_URL, headers=_headers(), json=payload)
                resp.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        data = resp.json()
        completion = data["choices"][0]["message"]["content"].strip()

        # Попытка выделить ссылки (url)
        import re
        urls = re.findall(r'https?://[^\s)]+', completion)
        # Альтернативный Citations-поиск
        meta_urls = []
        if "citations" in data["choices"][0]["message"]:
            # + Perplexity иногда возвращает раздел citations как список dict
            cits = data["choices"][0]["message"]["citations"]
            for cit in cits:
                if isinstance(cit, dict) and "url" in cit:
                    meta_urls.append(cit["url"])
        all_urls = sorted(list(set(urls + meta_urls)))
        return {
            "answer": completion,
            "sources": all_urls if return_sources else [],
            "raw": data,
        }

# Быстрая CLI-заглушка для теста
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "theory of resonance in ai"
    out = asyncio.run(perplexity_search(q))
    logger.info("-" * 40)
    logger.info("🧠 Perplexity Search Result:")
    logger.info(out["answer"])
    logger.info("-" * 40)
    logger.info("🔗 Links:")
    for link in out["sources"]:
        logger.info("  • %s", link)
