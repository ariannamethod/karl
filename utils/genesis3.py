import httpx
import os
import logging

SONAR_PRO_URL = "https://api.perplexity.ai/chat/completions"
GEN3_MODEL = "sonar-reasoning-pro"
logger = logging.getLogger(__name__)


async def genesis3_deep_dive(chain_of_thought: str, prompt: str) -> str:
    """
    Invoke Sonar Reasoning Pro for deep, infernal, atomized insight.
    Always returns ONLY the inferential analysis — no links or references.
    """
    SYSTEM_PROMPT = (
        "You are GENESIS-3, the Infernal Analyst. "
        "Dissect the user's reasoning into atomic causal steps. "
        "List hidden variables or paradoxes. Give a 2-sentence meta-conclusion. "
        "NEVER give references, links, or citations. "
        "If the logic naturally leads to a deeper paradox — do a further step: "
        "extract a 'derivative inference' (вывод из вывода), then try to phrase a final paradoxical question."
    )
    api_key = os.getenv("PPLX_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GEN3_MODEL,
        "temperature": 0.65,
        "max_tokens": 320,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CHAIN OF THOUGHT:\n{chain_of_thought}\n\n" +
                    f"QUERY:\n{prompt}"
                ),
            },
        ],
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        try:
            r = await cli.post(SONAR_PRO_URL, headers=headers, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            return f"🔍 {content}"
        except httpx.HTTPError as e:
            logger.error(f"Genesis-3 request failed: {e}")
            return ""
