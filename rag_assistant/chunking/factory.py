from .fixed_size import FixedSizeChunker
from .recursive import RecursiveCharacterChunker
from .structure_aware import MarkdownStructureChunker
from .sentence import SentenceChunker

class ChunkerFactory:
    _strategies = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveCharacterChunker,
        "markdown": MarkdownStructureChunker,
        "sentence": SentenceChunker
    }

    @classmethod
    def get_chunker(cls, strategy: str, **kwargs):
        chunker_cls = cls._strategies.get(strategy)
        if not chunker_cls:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        return chunker_cls(**kwargs)
