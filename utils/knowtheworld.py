"""Background task that immerses Indiana in daily world news."""

import os
import random
import asyncio
from datetime import datetime, timezone

import httpx
from openai import AsyncOpenAI

from .vectorstore import VectorStore
from .memory import MemoryManager


vector_store = VectorStore()
memory = MemoryManager(db_path="lighthouse_memory.db", vectorstore=vector_store)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def _location() -> str:
    """Attempt to determine the current city and country."""
    try:
        resp = httpx.get("https://ipapi.co/json", timeout=10)
        data = resp.json()
        city = data.get("city")
        country = data.get("country_name")
        if city and country:
            return f"{city}, {country}"
        if country:
            return country
    except Exception:
        pass
    return "your area"


async def _fetch_recent_messages(limit: int = 10) -> str:
    """Return last `limit` messages from memory as context."""
    cur = memory.db.execute(
        "SELECT query, response FROM memory ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    if not rows:
        return ""
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in rows)


async def _store_insight(text: str):
    """Store generated insight in Pinecone with tag #knowtheworld."""
    vector = await vector_store.embed_text(text)
    now = datetime.now(timezone.utc).isoformat()
    vector_store.index.upsert(
        [(f"know-{now}", vector, {"date": now, "tag": "#knowtheworld", "text": text})]
    )


async def _gather_news() -> str:
    """Use OpenAI to summarise local and world news."""
    loc = await _location()
    prompt = (
        "Provide a brief overview of today's news in "
        f"{loc}. Then add short summaries from Paris, Berlin, New York, Moscow, "
        "Amsterdam and other global headlines."
    )
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


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
    await know_the_world()
    while True:
        delay = random.uniform(0, 86400)
        await asyncio.sleep(delay)
        try:
            await know_the_world()
        except Exception:
            pass
        await asyncio.sleep(86400 - delay)

