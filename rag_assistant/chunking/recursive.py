from typing import List, Any
import hashlib
import re
from .models import BaseChunker, Chunk

class RecursiveCharacterChunker(BaseChunker):
    """
    Attempts to split on a list of characters in order (e.g. \n\n, \n, " ", "").
    This keeps semantically related text (like paragraphs) together.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, metadata: dict[str, Any]) -> List[Chunk]:
        raw_chunks = self._recursive_split(text, self.separators)

        # Now merge the small parts into actual chunks of roughly chunk_size
        final_chunks = []
        current_content = ""
        doc_hash = metadata.get("content_hash", hashlib.md5(text.encode()).hexdigest()[:8])

        for part in raw_chunks:
            if len(current_content) + len(part) <= self.chunk_size:
                current_content += part
            else:
                if current_content:
                    final_chunks.append(self._create_chunk(current_content, len(final_chunks), metadata, doc_hash))

                # Handle overlap: take a slice of the previous content
                if self.chunk_overlap > 0:
                    current_content = current_content[-self.chunk_overlap:] + part
                else:
                    current_content = part

        if current_content:
            final_chunks.append(self._create_chunk(current_content, len(final_chunks), metadata, doc_hash))

        return final_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        new_separators = separators[1:]

        parts = text.split(separator)
        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part + (separator if part != parts[-1] else ""))
            else:
                # Recurse with the next separator
                result.extend(self._recursive_split(part, new_separators))
        return result

    def _create_chunk(self, content: str, index: int, metadata: dict[str, Any], doc_hash: str) -> Chunk:
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": index,
            "strategy": "recursive"
        })
        return Chunk(
            id=f"{doc_hash}_{index}",
            content=content,
            metadata=chunk_metadata
        )
