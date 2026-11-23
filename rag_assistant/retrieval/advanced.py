from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from ..vectorstore.chroma import ChromaVectorStore
from ..chunking.models import Chunk

class AdvancedRetriever:
    def __init__(self, vector_store: ChromaVectorStore, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vstore = vector_store
        self.reranker = CrossEncoder(reranker_model)
        # We need all documents in memory for BM25
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []

    def _prepare_bm25(self):
        """Prepares BM25 index from all documents currently in Chroma."""
        all_docs = self.vstore.collection.get()
        self.corpus = all_docs['documents']
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Combines Vector Search and BM25 using Reciprocal Rank Fusion (RRF).
        alpha: weight for vector search (1.0 = vector only, 0.0 = BM25 only)
        """
        if self.bm25 is None:
            self._prepare_bm25()

        # 1. Vector Search
        v_results = self.vstore.search(query, top_k=top_k)

        # 2. BM25 Search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

        # 3. Simple Merge (In production, use weighted RRF)
        return self._merge_results(v_results, top_bm25_indices, bm25_scores)

    def _merge_results(self, v_results, bm25_indices, bm25_scores):
        # Implementation of RRF or weighted score merge
        # For simplicity, returning top vector results combined with reranking
        merged = []
        # ... logic to deduplicate and score ...
        return v_results # Placeholder for refined merge

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[str]:
        """Uses Cross-Encoder to re-score documents."""
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1]
        return [documents[i] for i in ranked_indices[:top_n]]
