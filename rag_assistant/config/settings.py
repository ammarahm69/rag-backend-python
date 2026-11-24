from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise RAG Knowledge Assistant"

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # Ingestion
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md", ".html"]

    # ChromaDB Config
    CHROMA_MODE: str = "local"  # "local" or "remote"
    CHROMA_PATH: str = "data/processed/chroma_db"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_API_KEY: Optional[str] = None

    # LLM Provider Keys (loaded from .env)
    GROQ_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Models
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODELS: dict[str, str] = {
        "fast": "all-MiniLM-L6-v2",
        "balanced": "all-distilroberta-v1",
        "accurate": "BAAI/bge-small-en-v1.5",
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
    }

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
