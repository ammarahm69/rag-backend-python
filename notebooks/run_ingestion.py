import sys
from pathlib import Path

# Add current dir to path to resolve imports
sys.path.append(str(Path(__file__).parent.parent))

from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.config.settings import settings

def main():
    pipeline = IngestionPipeline()
    docs = pipeline.run(settings.RAW_DATA_DIR)
    print(f"Ingested {len(docs)} documents.")
    for doc in docs:
        print(f"File: {doc.metadata['file_name']}, Hash: {doc.content_hash}")

if __name__ == "__main__":
    main()
