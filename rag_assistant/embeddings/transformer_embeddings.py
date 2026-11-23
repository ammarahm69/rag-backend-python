import os
import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder
from ..config.settings import settings

class TransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir or Path(settings.PROCESSED_DATA_DIR) / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Device selection: cuda > mps (mac) > cpu
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = SentenceTransformer(model_name, device=self.device)

        # Check if model requires prefixes (like BGE or E5)
        self.doc_prefix = ""
        self.query_prefix = ""
        if "bge" in model_name.lower():
            self.query_prefix = "Represent this sentence for searching relevant passages: "
        elif "e5" in model_name.lower():
            self.doc_prefix = "passage: "
            self.query_prefix = "query: "

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _get_cache_path(self, text: str) -> Path:
        # Hash of (model_name + text)
        hash_input = f"{self.model_name}_{text}"
        text_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.npy"

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed_with_cache(texts, is_query=False)

    def embed_query(self, text: str) -> np.ndarray:
        # Queries are usually not cached as they are dynamic
        prefixed_text = f"{self.query_prefix}{text}"
        return self.model.encode(
            [prefixed_text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )[0]

    def _embed_with_cache(self, texts: List[str], is_query: bool) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix

        results = [None] * len(texts)
        to_compute_indices = []
        to_compute_texts = []

        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                results[i] = np.load(cache_path)
            else:
                to_compute_indices.append(i)
                to_compute_texts.append(f"{prefix}{text}")

        if to_compute_texts:
            computed_embeddings = self.model.encode(
                to_compute_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True
            )

            for i, idx in enumerate(to_compute_indices):
                embedding = computed_embeddings[i]
                results[idx] = embedding
                # Save to cache
                np.save(self._get_cache_path(texts[idx]), embedding)

        return np.array(results)
