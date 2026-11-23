import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from typing import List, Any, Optional
import numpy as np
from ..embeddings.base import BaseEmbedder
from ..chunking.models import Chunk
from ..config.settings import settings

class ChromaVectorStore:
    def __init__(self, collection_name: str, embedder: BaseEmbedder):
        self.embedder = embedder

        if settings.CHROMA_MODE == "remote":
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT
            )
        else:
            self.client = chromadb.PersistentClient(
                path=str(settings.BASE_DIR / settings.CHROMA_PATH)
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk]):
        """Upserts chunks into the collection."""
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [c.id for c in chunks]

        # Embed using our custom embedder
        embeddings = self.embedder.embed_documents(texts)

        self.collection.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )

    def search(self, query: str, top_k: int = 5, where: Optional[dict] = None):
        """Semantic search with optional metadata filtering."""
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where
        )
        return results
