from pydantic import BaseModel, Field
from typing import Any, List, Optional
from abc import ABC, abstractmethod

class Chunk(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
    parent_id: Optional[str] = None

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> List[Chunk]:
        pass

    def get_stats(self, chunks: List[Chunk]):
        if not chunks:
            return {"count": 0}

        lengths = [len(c.content) for c in chunks]
        return {
            "count": len(chunks),
            "avg_chars": sum(lengths) / len(chunks),
            "min_chars": min(lengths),
            "max_chars": max(lengths)
        }
