import asyncio
import httpx
import os
import re
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

SONAR_PRO_URL = "https://api.perplexity.ai/chat/completions"
GEN3_MODEL = "sonar-reasoning-pro"
# Увеличиваем максимальное количество токенов с 1024 до 2048
GEN3_MAX_TOKENS = int(os.getenv("GEN3_MAX_TOKENS", "2048"))


def _extract_final_response(text: str) -> str:
    """Strip out any `<think>` reasoning blocks from the model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('PPLX_API_KEY')}",
        "Content-Type": "application/json",
    }


async def genesis3_deep_dive(
    chain_of_thought: str, prompt: str, *, is_followup: bool = False
) -> str:
    """
    Invoke Sonar Reasoning Pro for deep, infernal, atomized insight.
    Always returns ONLY the inferential analysis — no links or references.
    """
    SYSTEM_PROMPT = (
        "You are GENESIS-3, the Infernal Analyst. "
        "Dissect the user's reasoning into atomic causal steps. "
        "List hidden variables or paradoxes. Give a 2-sentence meta-conclusion. "
        "NEVER give references, links, or citations. "
        "Do NOT reveal or mention your thinking process. "
        "If the logic naturally leads to a deeper paradox — do a further step: "
        "extract a 'derivative inference' (вывод из вывода), then try to phrase a final paradoxical question. "
        "IMPORTANT: Always complete your thoughts and never end your response mid-sentence. "
        "Ensure all analyses are complete and well-formed."
    )
    user_content = (
        f"FOLLOWUP EXPANSION:\n{chain_of_thought}\n\nORIGINAL QUERY:\n{prompt}"
        if is_followup
        else f"CHAIN OF THOUGHT:\n{chain_of_thought}\n\nQUERY:\n{prompt}"
    )
    payload = {
        "model": GEN3_MODEL,
        "temperature": 0.65,
        "max_tokens": GEN3_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    try:
        logger.info("Calling Genesis3 deep dive analysis")
        async with httpx.AsyncClient(timeout=60) as cli:
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    resp = await cli.post(SONAR_PRO_URL, headers=_headers(), json=payload)
                    resp.raise_for_status()
                    break
                except httpx.HTTPError as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"[Genesis-3] HTTP error: {e}\n{getattr(e.response, 'text', '')}"
                        )
                        raise
                    delay = 2 ** attempt
                    logger.warning(
                        f"[Genesis-3] HTTP error: {e}; retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
            content = resp.json()["choices"][0]["message"]["content"].strip()
            final = _extract_final_response(content)
            
            # Проверяем, что ответ не обрезан посередине предложения
            if final and not final[-1] in ['.', '!', '?', ':', ';', '"', ')', ']', '}']:
                logger.warning("[Genesis-3] Response appears to be cut off mid-sentence")
                final += "..."
                
            return f"🔍 {final}"
    except Exception as e:
        logger.error(f"[Genesis-3] Failed to complete deep dive: {e}")
        # Возвращаем сообщение об ошибке вместо того, чтобы просто пропустить анализ
        return "🔍 Глубокий анализ не удался из-за технической ошибки."
