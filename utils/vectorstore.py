import os
from pinecone import Pinecone
from openai import AsyncOpenAI


class VectorStore:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Connect to index
        self.index_name = os.getenv("PINECONE_INDEX")
        self.index = self.pc.Index(self.index_name)

    async def embed_text(self, text: str):
        """Generate embeddings for text using OpenAI."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding

    async def store(self, id: str, text: str):
        """Store text embedding in Pinecone."""
        vector = await self.embed_text(text)
        self.index.upsert([(id, vector, {"text": text})])

    async def search(self, query: str, top_k: int = 5):
        """Search for similar texts."""
        query_vector = await self.embed_text(query)
        results = self.index.query(query_vector, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"]]
