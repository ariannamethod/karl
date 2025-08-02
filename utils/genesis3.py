import httpx
import os

SONAR_PRO_URL = "https://api.perplexity.ai/chat/completions"
GEN3_MODEL = "sonar-reasoning-pro"
PRO_HEADERS = {
    "Authorization": f"Bearer {os.getenv('PPLX_API_KEY')}",
    "Content-Type": "application/json"
}


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
    payload = {
        "model": GEN3_MODEL,
        "temperature": 0.65,
        "max_tokens": 320,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CHAIN OF THOUGHT:\n{chain_of_thought}"},
            {"role": "user", "content": f"QUERY:\n{prompt}"}
        ]
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(SONAR_PRO_URL, headers=PRO_HEADERS, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        return f"🔍 {content}"
