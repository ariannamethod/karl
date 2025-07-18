import random
import textwrap
from datetime import datetime, timezone

from openai import AsyncOpenAI

from .config import settings

# Genesis2 previously referenced a non-existent model name.  The
# correct OpenAI model identifier is simply "o3" as per the public
# documentation.  Using the wrong name resulted in model_not_found
# errors during runtime.
OPENAI_MODEL = "o3"
TIMEOUT = 25

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None


def _build_prompt(draft: str, user_prompt: str) -> list[dict[str, str]]:
    """Compose a short prompt for the intuition filter."""
    system_msg = textwrap.dedent(
        """
        You are GENESIS-2, the intuition filter for Indiana‐AM (an
        “Indiana Jones” archetype).  Return ONE short investigative twist
        (≤120 tokens) that deepens the current reasoning.  Do **NOT**
        repeat the draft; just add an angle, question or hidden variable.
        """
    ).strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"USER PROMPT >>> {user_prompt}"},
        {"role": "assistant", "content": f"DRAFT >>> {draft}"},
        {"role": "user", "content": "Inject the twist now:"},
    ]


async def _call_openai(messages: list[dict[str, str]]) -> str:
    """Send a prompt to OpenAI and return the response text."""
    if not client:
        raise RuntimeError("OpenAI client not configured")

    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.9,
        messages=messages,
        max_tokens=120,
        timeout=TIMEOUT,
    )
    return resp.choices[0].message.content.strip()


async def genesis2_filter(user_prompt: str, draft_reply: str) -> str:
    """Return a short twist or an empty string."""
    if random.random() < 0.10 or not client:
        return ""

    try:
        twist = await _call_openai(_build_prompt(draft_reply, user_prompt))
        return twist
    except Exception as e:
        print(f"[Genesis-2] GPT fail {e}  @ {datetime.now(timezone.utc).isoformat()}")
        return ""


async def assemble_final_reply(user_prompt: str, indiana_draft: str) -> str:
    """Weave the twist into the final reply."""
    twist = await genesis2_filter(user_prompt, indiana_draft)
    if twist:
        final = f"{indiana_draft}\n\n🜂 Investigative Twist → {twist}"
    else:
        final = indiana_draft
    return final
