import asyncio
import logging
import math
import time
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
    async def store(
        self,
        id: str,
        text: str,
        *,
        user_id: str | None = None,
        metadata: Dict | None = None,
    ):
        """Store text with optional user metadata and extra metadata."""
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

        async def store(
            self,
            id: str,
            text: str,
            *,
            user_id: str | None = None,
            metadata: Dict | None = None,
        ):
            vector = await self.embed_text(text)
            for attempt in range(3):
                try:
                    meta = {"text": text}
                    if user_id:
                        meta["user"] = user_id
                    if metadata:
                        meta.update(metadata)
                    await asyncio.to_thread(
                        self.index.upsert, vectors=[(id, vector, meta)]
                    )
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
                    results = await asyncio.to_thread(self.index.query, **params)
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
        # store as OrderedDict {id: (text, embedding, user_id, metadata)}
        self._store: "OrderedDict[str, Tuple[str, List[float] | None, str | None, Dict | None]]" = OrderedDict()
        self._embed_cache: Dict[str, List[float]] = {}
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.client = (
            AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            if settings.OPENAI_API_KEY
            else None
        )
        self._save_task: asyncio.Task | None = None
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data:
                        tup = tuple(v)
                        if len(tup) == 4:
                            text, emb, uid, meta = tup
                            self._store[k] = (text, emb, uid, meta)
                            if emb is not None:
                                self._embed_cache[text] = emb
                        else:  # backward compatibility
                            text, uid, meta = tup
                            self._store[k] = (text, None, uid, meta)
            except Exception as e:
                logger.error(
                    "Failed to load vector store from %s: %s", self.persist_path, e
                )
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

    async def _generate_embedding(self, text: str) -> List[float]:
        if self.client:
            response = await self.client.embeddings.create(
                model=self.embed_model, input=text
            )
            return response.data[0].embedding
        # Fallback simple embedding: character frequency vector
        vec = [0.0] * 26
        for ch in text.lower():
            idx = ord(ch) - 97
            if 0 <= idx < 26:
                vec[idx] += 1.0
        return vec

    async def embed_text(self, text: str) -> List[float]:
        if text in self._embed_cache:
            return self._embed_cache[text]
        emb = await self._generate_embedding(text)
        self._embed_cache[text] = emb
        return emb

    async def store(
        self,
        id: str,
        text: str,
        *,
        user_id: str | None = None,
        metadata: Dict | None = None,
    ):
        vector = await self.embed_text(text)
        if id in self._store:
            self._store.move_to_end(id)
        self._store[id] = (text, vector, user_id, metadata)
        # Evict the oldest entry when exceeding ``max_size``
        if self.max_size is not None and len(self._store) > self.max_size:
            self._store.popitem(last=False)
        # Persist asynchronously if configured
        if self.persist_path:
            if self._save_task and not self._save_task.done():
                self._save_task.cancel()
            self._save_task = asyncio.create_task(asyncio.to_thread(self._save))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        max_time: float | None = None,
        max_docs: int | None = None,
    ) -> List[str]:
        query_vec = await self.embed_text(query)
        query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1.0
        scored: List[Tuple[str, float]] = []
        start = time.time()
        processed = 0
        for key, value in list(self._store.items()):
            if max_docs is not None and processed >= max_docs:
                break
            text, emb, uid, meta = value if len(value) == 4 else (
                value[0], None, value[1], value[2]
            )
            if user_id and uid != user_id:
                continue
            if emb is None:
                emb = await self.embed_text(text)
                self._store[key] = (text, emb, uid, meta)
            dot = sum(a * b for a, b in zip(query_vec, emb))
            emb_norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            score = dot / (query_norm * emb_norm)
            scored.append((text, score))
            processed += 1
            if max_time is not None and time.time() - start >= max_time:
                break
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:top_k]]


def create_vector_store(max_size: int | None = None, persist_path: str | None = None) -> BaseVectorStore:
    if max_size is None:
        max_size = settings.VECTOR_STORE_MAX_SIZE
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
