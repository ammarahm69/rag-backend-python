import sys
from pathlib import Path

# Add current dir to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.config.settings import settings
from rag_assistant.chunking.factory import ChunkerFactory

def main():
    # 1. Ingest
    pipeline = IngestionPipeline()
    docs = pipeline.run(settings.RAW_DATA_DIR)

    # 2. Chunking demo with SMALLER chunk size to see actual splits
    strategies = [
        ("fixed", {"chunk_size": 100, "chunk_overlap": 20}),
        ("recursive", {"chunk_size": 100, "chunk_overlap": 20}),
        ("markdown", {}),
        ("sentence", {"sentences_per_chunk": 1})
    ]

    for doc in docs:
        print(f"\n==========================================")
        print(f"Document: {doc.metadata['file_name']}")
        print(f"==========================================")

        for strategy_name, kwargs in strategies:
            try:
                chunker = ChunkerFactory.get_chunker(strategy_name, **kwargs)
                chunks = chunker.chunk(doc.content, doc.metadata)
                stats = chunker.get_stats(chunks)
                print(f"\n--- Strategy: {strategy_name} ---")
                print(f"Stats: {stats}")
                print("First 2 Chunks:")
                for c in chunks[:2]:
                    print(f"  [ID: {c.id}] -> {repr(c.content[:60])}...")
            except Exception as e:
                print(f"Strategy: {strategy_name:15} | Failed: {e}")

if __name__ == "__main__":
    main()
