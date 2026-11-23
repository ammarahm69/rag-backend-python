import hashlib
from typing import Optional, Dict, Any
from ..vectorstore.chroma import ChromaVectorStore
from ..embeddings.base import BaseEmbedder

class SemanticCache:
    def __init__(self, embedder: BaseEmbedder, threshold: float = 0.95):
        # We use a separate collection for the cache
        from chromadb import PersistentClient
        from ..config.settings import settings

        self.embedder = embedder
        self.threshold = threshold
        self.client = PersistentClient(path=str(settings.PROCESSED_DATA_DIR / "chroma_db"))
        self.collection = self.client.get_or_create_collection(
            name="semantic_cache",
            metadata={"hnsw:space": "cosine"}
        )

    def get(self, query: str) -> Optional[str]:
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1,
            include=["documents", "distances", "metadatas"]
        )

        if results['documents'] and results['distances'][0]:
            # Cosine distance: 0 is identical, 2 is opposite.
            # We want high similarity (low distance)
            distance = results['distances'][0][0]
            similarity = 1 - (distance / 2) # Normalized to 0-1

            if similarity >= self.threshold:
                return results['documents'][0][0]

        return None

    def set(self, query: str, answer: str):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        query_embedding = self.embedder.embed_query(query)

        self.collection.upsert(
            ids=[query_hash],
            documents=[answer],
            embeddings=[query_embedding.tolist()],
            metadatas=[{"query": query}]
        )
