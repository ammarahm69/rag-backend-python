from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import shutil
from pathlib import Path

from ..config.settings import settings
from ..ingestion.pipeline import IngestionPipeline
from ..chunking.factory import ChunkerFactory
from ..embeddings.transformer_embeddings import TransformerEmbedder
from ..vectorstore.chroma import ChromaVectorStore
from ..retrieval.advanced import AdvancedRetriever
from ..generation.llm_client import OpenAICompatibleClient
from ..generation.generator import RAGGenerator
from ..optimization.cache import SemanticCache
from ..observability.tracer import LatencyTracer

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise RAG Knowledge Assistant API with Citations and Advanced Retrieval",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc alternative
)

# Global instances initialized on startup
embedder = None
vector_store = None
pipeline = None
cache = None

@app.on_event("startup")
def startup_event():
    global embedder, vector_store, pipeline, cache
    embedder = TransformerEmbedder(model_name=settings.DEFAULT_EMBEDDING_MODEL)
    vector_store = ChromaVectorStore(collection_name="enterprise_knowledge", embedder=embedder)
    pipeline = IngestionPipeline()
    cache = SemanticCache(embedder=embedder)

# Pydantic Schemas for API Request/Response
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the remote work policy?")
    top_k: int = Field(5, example=5)
    use_cache: bool = Field(True, description="Enable semantic cache")

class QueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool
    sources: List[Dict[str, Any]]
    latency: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    version: str

# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/ingest", tags=["Data Ingestion"])
def ingest_file(file: UploadFile = File(...)):
    """Upload a document (PDF, TXT, MD, HTML) to ingest into the RAG vector store."""
    file_path = settings.RAW_DATA_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = pipeline.run(settings.RAW_DATA_DIR)

        # Chunk and Add to ChromaDB
        chunker = ChunkerFactory.get_chunker("recursive", chunk_size=500, chunk_overlap=50)
        total_chunks = 0
        for doc in docs:
            chunks = chunker.chunk(doc.content, doc.metadata)
            vector_store.add_chunks(chunks)
            total_chunks += len(chunks)

        return {"filename": file.filename, "status": "success", "chunks_added": total_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["RAG Query"])
def query_rag(request: QueryRequest):
    """Query the RAG pipeline with optional semantic caching and full latency breakdown."""
    tracer = LatencyTracer()

    # 1. Check Cache
    if request.use_cache:
        with tracer.track("semantic_cache_check"):
            cached_answer = cache.get(request.query)
            if cached_answer:
                return QueryResponse(
                    query=request.query,
                    answer=cached_answer,
                    cached=True,
                    sources=[],
                    latency=tracer.get_summary()
                )

    # 2. Retrieval
    with tracer.track("vector_retrieval"):
        search_results = vector_store.search(request.query, top_k=request.top_k)

    # 3. Generation using Groq (Free & Fast) - Config from .env
    with tracer.track("llm_generation"):
        client = OpenAICompatibleClient(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b"
        )
        generator = RAGGenerator(llm_client=client)
        try:
            # CLEAR THE CACHE FOR THIS QUERY SO WE CAN SEE THE NEW MODEL ATTEMPT
            answer = generator.answer(request.query, search_results)
        except Exception as e:
            answer = f"Error communicating with Groq: {str(e)}"

    # 4. Save to Cache
    if request.use_cache and answer:
        cache.set(request.query, answer)

    # Format sources
    sources = []
    if search_results.get('metadatas') and search_results['metadatas'][0]:
        for meta in search_results['metadatas'][0]:
            sources.append({"source": meta.get("file_name"), "doc_type": meta.get("doc_type")})

    return QueryResponse(
        query=request.query,
        answer=answer,
        cached=False,
        sources=sources,
        latency=tracer.get_summary()
    )
