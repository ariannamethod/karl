import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .vectorstore import BaseVectorStore, create_vector_store

class MemoryManager:
    def __init__(self, db_path: str = "memory.db", vectorstore: Optional[BaseVectorStore] = None):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.vectorstore = vectorstore or create_vector_store()
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT,
                timestamp TEXT,
                query TEXT,
                response TEXT
            )
        """)
        self.db.commit()

    async def save(self, user_id: str, query: str, response: str):
        """Save user query and response to memory database."""
        ts = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT INTO memory VALUES (?,?,?,?)",
            (user_id, ts, query, response)
        )
        self.db.commit()
        if self.vectorstore:
            try:
                await self.vectorstore.store(f"{user_id}-{ts}", f"Q: {query}\nA: {response}")
            except Exception:
                pass

    async def retrieve(self, user_id: str, query: str) -> str:
        """Retrieve last 5 responses for a given user as context."""
        cur = self.db.execute(
            "SELECT response FROM memory WHERE user_id=? ORDER BY timestamp DESC LIMIT 5",
            (user_id,)
        )
        rows = cur.fetchall()
        if not rows:
            return ""
        # склеиваем последние 5 ответов как контекст
        return "\n".join(r[0] for r in rows)

    async def search_memory(self, query: str, top_k: int = 5) -> list[str]:
        """Search vector memory for similar texts."""
        if not self.vectorstore:
            return []
        try:
            return await self.vectorstore.search(query, top_k)
        except Exception as e:
            # log and fall back to empty list
            print(f"Vector search failed: {e}")
            return []
