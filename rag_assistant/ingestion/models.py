from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any
import hashlib

class Document(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str

    @classmethod
    def create(cls, content: str, metadata: dict[str, Any]):
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["content_hash"] = content_hash
        return cls(content=content, metadata=metadata, content_hash=content_hash)
