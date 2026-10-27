from typing import Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from backend.config import get_settings


class QdrantClientManager:
    """
    Manages Qdrant client connections (HTTP or in-memory for testing).
    """

    _sync_client: Optional[QdrantClient] = None
    _async_client: Optional[AsyncQdrantClient] = None

    @classmethod
    def get_sync_client(cls) -> QdrantClient:
        if cls._sync_client is None:
            settings = get_settings()
            if settings.QDRANT_URL == ":memory:":
                cls._sync_client = QdrantClient(location=":memory:")
            else:
                cls._sync_client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=30,
                )
        return cls._sync_client

    @classmethod
    def get_async_client(cls) -> AsyncQdrantClient:
        if cls._async_client is None:
            settings = get_settings()
            if settings.QDRANT_URL == ":memory:":
                cls._async_client = AsyncQdrantClient(location=":memory:")
            else:
                cls._async_client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=30,
                )
        return cls._async_client

    @classmethod
    def reset(cls) -> None:
        cls._sync_client = None
        cls._async_client = None
