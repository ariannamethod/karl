import asyncio
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import os
import json
import atexit
from collections import OrderedDict

from openai import AsyncOpenAI
try:  # Optional dependency
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - optional
    Pinecone = None

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


if Pinecone:
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
else:  # pragma: no cover - optional
    RemoteVectorStore = None


class LocalVectorStore(BaseVectorStore):
    def __init__(self, max_size: int | None = 1000, persist_path: str | None = None):
        """In-memory vector store with optional size limit and persistence.

        Parameters
        ----------
        max_size:
            Maximum number of entries to retain. A sane default of ``1000`` is
            used to avoid unbounded memory growth. Pass ``None`` to disable the
            limit entirely.
        persist_path:
            Optional path to persist the store on shutdown.
        """
        self.max_size = max_size
        self.persist_path = persist_path
        # store as OrderedDict {id: (text, user_id)} to track insertion order
        self._store: "OrderedDict[str, Tuple[str, str | None]]" = OrderedDict()
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._store = OrderedDict((k, tuple(v)) for k, v in data)
            except Exception as e:
                logger.error("Failed to load vector store from %s: %s", self.persist_path, e)
        if self.persist_path:
            atexit.register(self._save)

    def _save(self):
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(list(self._store.items()), f)
        except Exception as e:
            logger.error("Failed to persist vector store: %s", e)

    async def store(self, id: str, text: str, *, user_id: str | None = None):
        if id in self._store:
            self._store.move_to_end(id)
        self._store[id] = (text, user_id)
        if self.max_size is not None and len(self._store) > self.max_size:
            self._store.popitem(last=False)

    async def search(self, query: str, top_k: int = 5, *, user_id: str | None = None) -> List[str]:
        scored: List[Tuple[str, float]] = []
        for text_id, (text, uid) in self._store.items():
            if user_id and uid != user_id:
                continue
            score = SequenceMatcher(None, query.lower(), text.lower()).ratio()
            scored.append((text, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:top_k]]


def create_vector_store(max_size: int | None = 1000, persist_path: str | None = None) -> BaseVectorStore:
    if (
        Pinecone
        and settings.OPENAI_API_KEY
        and settings.PINECONE_API_KEY
        and RemoteVectorStore is not None
    ):
        try:
            return RemoteVectorStore()
        except Exception as e:  # pragma: no cover - network
            logger.error("Failed to initialise remote vector store: %s", e)
    logger.warning("Using local vector store fallback")
    return LocalVectorStore(max_size=max_size, persist_path=persist_path)
