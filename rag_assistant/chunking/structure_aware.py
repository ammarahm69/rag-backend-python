import re
import hashlib
from typing import List, Any
from .models import BaseChunker, Chunk

class MarkdownStructureChunker(BaseChunker):
    """
    Splits document based on Markdown headers (#, ##, ###).
    """
    def chunk(self, text: str, metadata: dict[str, Any]) -> List[Chunk]:
        # Handle line endings uniformly
        lines = text.splitlines()
        chunks_data = []
        current_header = "Intro"
        current_lines = []

        doc_hash = metadata.get("content_hash", hashlib.md5(text.encode()).hexdigest()[:8])

        header_pattern = re.compile(r'^(#+\s+.*)')

        for line in lines:
            line_str = line.strip()
            if header_pattern.match(line_str):
                # Save previous chunk if it has content
                if current_lines:
                    chunks_data.append((current_header, "\n".join(current_lines)))
                current_header = line_str
                current_lines = []
            else:
                if line_str:
                    current_lines.append(line_str)

        # Add the last chunk
        if current_lines:
            chunks_data.append((current_header, "\n".join(current_lines)))

        chunks = []
        for i, (header, content) in enumerate(chunks_data):
            if not content.strip(): continue

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "header": header,
                "strategy": "markdown_structure"
            })

            chunks.append(Chunk(
                id=f"{doc_hash}_{i}",
                content=f"{header}\n{content.strip()}",
                metadata=chunk_metadata
            ))

        return chunks
