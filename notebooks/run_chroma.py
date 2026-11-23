import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.chunking.factory import ChunkerFactory
from rag_assistant.embeddings.transformer_embeddings import TransformerEmbedder
from rag_assistant.vectorstore.chroma import ChromaVectorStore
from rag_assistant.config.settings import settings

def main():
    # 1. Ingest
    print("--- Ingesting ---")
    pipeline = IngestionPipeline()
    docs = pipeline.run(settings.RAW_DATA_DIR)

    # 2. Setup Embedder and VectorStore
    print("--- Initializing Store ---")
    embedder = TransformerEmbedder(model_name=settings.DEFAULT_EMBEDDING_MODEL)
    vstore = ChromaVectorStore(collection_name="enterprise_knowledge", embedder=embedder)

    # 3. Chunk and Store
    print("--- Chunking and Indexing ---")
    chunker = ChunkerFactory.get_chunker("recursive", chunk_size=300, chunk_overlap=50)

    for doc in docs:
        chunks = chunker.chunk(doc.content, doc.metadata)
        vstore.add_chunks(chunks)
        print(f"Indexed {len(chunks)} chunks from {doc.metadata['file_name']}")

    # 4. Simple Test Query
    print("\n--- Testing Search ---")
    query = "How many days can I work remotely?"
    results = vstore.search(query, top_k=2)

    print(f"Query: {query}")
    for doc in results['documents'][0]:
        print(f"Found: {doc[:100]}...")

if __name__ == "__main__":
    main()
