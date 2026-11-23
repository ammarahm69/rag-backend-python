from typing import List, Any
import hashlib
from .models import BaseChunker, Chunk

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any]) -> List[Chunk]:
        chunks = []
        start = 0
        doc_hash = metadata.get("content_hash", hashlib.md5(text.encode()).hexdigest()[:8])

        while start < len(text):
            end = start + self.chunk_size
            content = text[start:end]

            chunk_id = f"{doc_hash}_{len(chunks)}"
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "chunk_start": start,
                "chunk_end": end,
                "strategy": "fixed_size"
            })

            chunks.append(Chunk(
                id=chunk_id,
                content=content,
                metadata=chunk_metadata
            ))

            start += (self.chunk_size - self.chunk_overlap)

        return chunks
