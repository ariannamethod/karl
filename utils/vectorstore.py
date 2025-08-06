import os
import asyncio
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from openai import AsyncOpenAI
from pinecone import Pinecone

from .config import settings

logger = logging.getLogger(__name__)


class BaseVectorStore:
    async def store(self, id: str, text: str, *, user_id: str | None = None):
        """Store text with optional user metadata."""
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 5, *, user_id: str | None = None) -> List[str]:
        """Return texts most similar to the query. If ``user_id`` is provided,
        restrict results to that user."""
        raise NotImplementedError


class RemoteVectorStore(BaseVectorStore):
    def __init__(self):
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX
        self.index = self.pc.Index(self.index_name)

    async def embed_text(self, text: str) -> List[float]:
        for attempt in range(3):
            try:
                response = await self.client.embeddings.create(
                    model=self.embed_model,
                    input=text,
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error("Embed attempt %s failed: %s", attempt + 1, e)
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def store(self, id: str, text: str, *, user_id: str | None = None):
        vector = await self.embed_text(text)
        for attempt in range(3):
            try:
                metadata = {"text": text}
                if user_id:
                    metadata["user"] = user_id
                self.index.upsert(vectors=[(id, vector, metadata)])
                return
            except Exception as e:
                logger.error("Pinecone upsert attempt %s failed: %s", attempt + 1, e)
                if attempt == 2:
                    return
                await asyncio.sleep(2 ** attempt)

    async def search(self, query: str, top_k: int = 5, *, user_id: str | None = None) -> List[str]:
        query_vector = await self.embed_text(query)
        for attempt in range(3):
            try:
                params = dict(vector=query_vector, top_k=top_k, include_metadata=True)
                if user_id:
                    params["filter"] = {"user": {"$eq": user_id}}
                results = self.index.query(**params)
                return [m["metadata"]["text"] for m in results["matches"]]
            except Exception as e:
                logger.error("Pinecone query attempt %s failed: %s", attempt + 1, e)
                if attempt == 2:
                    return []
                await asyncio.sleep(2 ** attempt)
        return []


class LocalVectorStore(BaseVectorStore):
    def __init__(self):
        # store as {id: (text, user_id)}
        self._store: Dict[str, Tuple[str, str | None]] = {}

    async def store(self, id: str, text: str, *, user_id: str | None = None):
        self._store[id] = (text, user_id)

    async def search(self, query: str, top_k: int = 5, *, user_id: str | None = None) -> List[str]:
        scored: List[Tuple[str, float]] = []
        for text_id, (text, uid) in self._store.items():
            if user_id and uid != user_id:
                continue
            score = SequenceMatcher(None, query.lower(), text.lower()).ratio()
            scored.append((text, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:top_k]]


def create_vector_store() -> BaseVectorStore:
    if settings.OPENAI_API_KEY and settings.PINECONE_API_KEY:
        try:
            return RemoteVectorStore()
        except Exception as e:
            logger.error("Failed to initialise remote vector store: %s", e)
    logger.warning("Using local vector store fallback")
    return LocalVectorStore()
