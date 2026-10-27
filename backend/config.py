from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("*", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    APP_ENV: str = "development"
    APP_NAME: str = "financial-rag"
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgrespassword@localhost:5432/financial_rag"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "financial_documents"

    # Translation LLMs (GPT-4o primary, Gemini fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_TRANSLATION_MODEL: str = "gpt-4o"
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_TRANSLATION_MODEL: str = "gemini-3.5-flash-lite"

    # English Embeddings (Sentence Transformers)
    ENGLISH_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    ENGLISH_EMBEDDING_REVISION: Optional[str] = None
    ENGLISH_EMBEDDING_DIMENSION: Optional[int] = None

    # Arabic Embeddings (Sentence Transformers)
    ARABIC_EMBEDDING_MODEL: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
    ARABIC_EMBEDDING_REVISION: Optional[str] = None
    ARABIC_EMBEDDING_DIMENSION: Optional[int] = None

    # Azure AI Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = None
    AZURE_DOCUMENT_INTELLIGENCE_KEY: Optional[str] = None

    # Background Tasks / Redis / Celery
    USE_CELERY: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage & Upload limits
    PRIVATE_STORAGE_ROOT: str = "./private_storage"
    MAX_UPLOAD_SIZE_MB: int = 50

    # Chunking limits
    CHUNK_SIZE_TOKENS: int = 384
    CHUNK_OVERLAP_TOKENS: int = 48
    PROCESSING_TIMEOUT_MINUTES: int = 30

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def storage_path(self) -> Path:
        p = Path(self.PRIVATE_STORAGE_ROOT).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache()
def get_settings() -> Settings:
    return Settings()
