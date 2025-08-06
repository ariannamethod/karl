import asyncio
import aiosqlite
from datetime import datetime, timezone
from typing import Optional

from .vectorstore import BaseVectorStore, create_vector_store

class MemoryManager:
    def __init__(self, db_path: str = "memory.db", vectorstore: Optional[BaseVectorStore] = None):
        self.db_path = db_path
        self.vectorstore = vectorstore or create_vector_store()

    async def _init_db(self, db: aiosqlite.Connection):
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT,
                timestamp TEXT,
                query TEXT,
                response TEXT
            )
            """
        )
        await db.commit()

    async def save(self, user_id: str, query: str, response: str):
        """Save user query and response to memory database and vector store."""
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await self._init_db(db)
            await db.execute(
                "INSERT INTO memory VALUES (?,?,?,?)",
                (user_id, ts, query, response),
            )
            await db.commit()
        if self.vectorstore:
            async def _store_vector():
                try:
                    await self.vectorstore.store(
                        f"{user_id}-{ts}",
                        f"Q: {query}\nA: {response}",
                        user_id=user_id,
                    )
                except Exception:
                    pass

            asyncio.create_task(_store_vector())

    async def retrieve(self, user_id: str, query: str) -> str:
        """Retrieve last 5 responses for a given user as context."""
        async with aiosqlite.connect(self.db_path) as db:
            await self._init_db(db)
            async with db.execute(
                "SELECT response FROM memory WHERE user_id=? ORDER BY timestamp DESC LIMIT 5",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        if not rows:
            return ""
        # склеиваем последние 5 ответов как контекст
        return "\n".join(r[0] for r in rows)

    async def search_memory(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        """Search vector memory for similar texts belonging to the given user."""
        if not self.vectorstore:
            return []
        try:
            return await self.vectorstore.search(query, top_k, user_id=user_id)
        except Exception as e:
            # log and fall back to empty list
            print(f"Vector search failed: {e}")
            return []

    async def last_response(self, user_id: str) -> str:
        """Return the most recent response for the given user."""
        async with aiosqlite.connect(self.db_path) as db:
            await self._init_db(db)
            async with db.execute(
                "SELECT response FROM memory WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
                (user_id,),
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else ""
