import uuid
from .vectorstore import create_vector_store


class KarlVectorEngine:
    """Minimal vector engine storing documents in Karl's vector store."""

    def __init__(self):
        self.store = create_vector_store()

    async def add_memory(self, identifier: str, text: str, *, role: str | None = None):
        """Store ``text`` in the vector store with a unique identifier."""
        uid = f"{identifier}-{uuid.uuid4().hex}"
        try:
            await self.store.store(uid, text, user_id=role)
        except Exception:
            pass
