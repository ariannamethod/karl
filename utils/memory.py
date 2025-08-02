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
        """Save user query and response to memory database and vector store."""
        ts = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT INTO memory VALUES (?,?,?,?)",
            (user_id, ts, query, response)
        )
        self.db.commit()
        if self.vectorstore:
            try:
                await self.vectorstore.store(
                    f"{user_id}-{ts}",
                    f"Q: {query}\nA: {response}",
                    user_id=user_id,
                )
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
        cur = self.db.execute(
            "SELECT response FROM memory WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
        return row[0] if row else ""

    async def retrieve_context_around(self, user_id: str, snippet: str, radius: int = 5) -> str:
        """Return ``2*radius+1`` responses surrounding the one matching ``snippet``."""
        cur = self.db.execute(
            "SELECT rowid FROM memory WHERE user_id=? AND response LIKE ? ORDER BY timestamp DESC LIMIT 1",
            (user_id, f"%{snippet}%"),
        )
        row = cur.fetchone()
        if not row:
            return ""
        rowid = row[0]
        start = max(1, rowid - radius)
        end = rowid + radius
        cur = self.db.execute(
            "SELECT response FROM memory WHERE user_id=? AND rowid BETWEEN ? AND ? ORDER BY rowid",
            (user_id, start, end),
        )
        rows = cur.fetchall()
        return "\n".join(r[0] for r in rows)

    async def check_vector_continuity(self):
        """Ensure the vector store is reachable."""
        if not self.vectorstore:
            print("Vector store not configured")
            return
        try:
            await self.vectorstore.search("ping", top_k=1)
            print("Vector memory reachable")
        except Exception as e:
            print(f"Vector memory check failed: {e}")
