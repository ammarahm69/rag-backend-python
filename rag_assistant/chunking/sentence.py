from typing import List, Any
import hashlib
import re
from .models import BaseChunker, Chunk

class SentenceChunker(BaseChunker):
    """
    Splits text into chunks of sentences.
    Uses simple regex for sentence boundaries (.!?).
    """
    def __init__(self, sentences_per_chunk: int = 3):
        self.sentences_per_chunk = sentences_per_chunk

    def chunk(self, text: str, metadata: dict[str, Any]) -> List[Chunk]:
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        doc_hash = metadata.get("content_hash", hashlib.md5(text.encode()).hexdigest()[:8])

        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.sentences_per_chunk]
            content = " ".join(chunk_sentences)

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i // self.sentences_per_chunk,
                "strategy": "sentence_based"
            })

            chunks.append(Chunk(
                id=f"{doc_hash}_{i}",
                content=content,
                metadata=chunk_metadata
            ))

        return chunks
