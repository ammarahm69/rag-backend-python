import sys
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag_assistant.embeddings.transformer_embeddings import TransformerEmbedder
from rag_assistant.config.settings import settings

def benchmark_model(model_name: str, sample_texts: List[str]):
    print(f"\nBenchmarking: {model_name}")
    start = time.time()
    embedder = TransformerEmbedder(model_name=model_name)
    embeddings = embedder.embed_documents(sample_texts)
    duration = time.time() - start

    print(f"Dimension: {embedder.dimension}")
    print(f"Time for {len(sample_texts)} texts: {duration:.4f}s")
    print(f"Throughput: {len(sample_texts)/duration:.2f} texts/sec")

    return duration

if __name__ == "__main__":
    texts = ["This is a test document.", "RAG enhances LLMs."] * 100

    for label, model in settings.EMBEDDING_MODELS.items():
        try:
            benchmark_model(model, texts)
        except Exception as e:
            print(f"Model {model} failed: {e}")
