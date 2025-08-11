"""Background task that immerses Indiana in daily world news."""

import random
import asyncio
import logging
from datetime import datetime, timezone

import httpx
from openai import AsyncOpenAI

from .vectorstore import create_vector_store
from .memory import MemoryManager
from .config import settings


vector_store = create_vector_store()
memory = MemoryManager(db_path="lighthouse_memory.db", vectorstore=vector_store)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
logger = logging.getLogger(__name__)


async def _location() -> str:
    """Attempt to determine the current city and country."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://ipapi.co/json", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            city = data.get("city")
            country = data.get("country_name")
            if city and country:
                return f"{city}, {country}"
            if country:
                return country
    except httpx.HTTPError as exc:
        logger.warning("Failed to determine location: %s", exc)
    return "your area"


async def _fetch_recent_messages(limit: int = 10) -> str:
    """Return last `limit` messages from memory as context."""
    rows = await memory.recent_messages(limit)
    if not rows:
        return ""
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in rows)


async def _store_insight(text: str):
    """Store generated insight in Pinecone with tag #knowtheworld."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        await vector_store.store(f"know-{now}", text, user_id="world")
    except Exception as exc:
        logger.warning("Failed to store insight: %s", exc)


async def _gather_news() -> str:
    """Use OpenAI to summarise local and world news."""
    loc = await _location()
    prompt = (
        "Provide a brief overview of today's news in "
        f"{loc}. Then add short summaries from Paris, Berlin, New York, Moscow, "
        "Amsterdam and other global headlines."
    )
    if client:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    return "No news available while offline."


async def _analyse_and_store(news: str):
    """Derive insight from news and recent chats and store the result."""
    context = await _fetch_recent_messages()
    prompt = (
        "You read the following news summary:\n" + news +
        "\nUsing the stored conversations below, find connections "
        "between world events and ongoing discussions. Build chains A→B→C "
        "leading to a paradoxical conclusion and reveal what is hidden.\n" +
        context
    )
    if client:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        insight = resp.choices[0].message.content.strip()
        await _store_insight(insight)


async def know_the_world():
    """Perform one cycle of world-awareness update."""
    news = await _gather_news()
    await _analyse_and_store(news)


async def start_world_task():
    """Run know_the_world daily at a random time."""
    try:
        await know_the_world()
        while True:
            delay = random.uniform(0, 86400)
            await asyncio.sleep(delay)
            try:
                await know_the_world()
            except Exception as exc:
                logger.warning("World awareness update failed: %s", exc)
            await asyncio.sleep(86400 - delay)
    except asyncio.CancelledError:
        logger.info("World task cancelled")
