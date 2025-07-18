import os
import asyncio
from datetime import datetime, timezone
from openai import AsyncOpenAI
from .vectorstore import VectorStore

vector_store = VectorStore()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def _fetch_last_day():
    """Return the date string of the last daily log if present."""
    try:
        result = vector_store.index.fetch(ids=["last-daily"])
        data = result.get("vectors", {}).get("last-daily")
        if data:
            return data.get("metadata", {}).get("date")
    except Exception:
        return None
    return None

async def _store_last_day(date: str, text: str):
    vector = await vector_store.embed_text(text)
    vector_store.index.upsert([
        (f"daily-{date}", vector, {"date": date, "text": text}),
        ("last-daily", vector, {"date": date})
    ])

async def default_reflection() -> str:
    """Generate a short daily reflection via OpenAI."""
    prompt = (
        "Summarise today's experiences in a couple of sentences and add your own "
        "thoughts about the day. Mention no personal user data."
    )
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

async def ensure_daily_entry(reflection_fn=default_reflection):
    """Create a daily log entry in Pinecone if one hasn't been stored today."""
    today = datetime.now(timezone.utc).date().isoformat()
    last = await _fetch_last_day()
    if last != today:
        text = await reflection_fn()
        await _store_last_day(today, text)
        return text
    return None

async def init_vector_memory():
    """Check and report the date of the last stored daily entry."""
    last = await _fetch_last_day()
    if last:
        print(f"Last daily log stored on {last}")
    else:
        print("No daily log found in vector memory")

async def start_daily_task():
    """Background task that ensures a daily entry every 24 hours."""
    while True:
        try:
            await ensure_daily_entry()
        except Exception:
            pass
        await asyncio.sleep(86400)
