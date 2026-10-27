import asyncio
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

TEST_DB_PATH = Path(__file__).parent / "test_financial_rag.db"
TEST_DB_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"

# Set test environment BEFORE importing backend modules
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["OPENAI_API_KEY"] = "test_openai_key"
os.environ["QDRANT_URL"] = ":memory:"
os.environ["CELERY_BROKER_URL"] = ""

from typing import AsyncGenerator
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from backend.config import Settings, get_settings
# Clear cache to guarantee test settings are loaded
get_settings.cache_clear()
from backend.constants import DocumentStatus, JobStatus
from backend.database.models import Base
from backend.database.session import get_db, async_session_factory, engine
from backend.main import create_app
from backend.ingestion.storage.private_store import PrivateStorageManager
from backend.ingestion.vector_store.qdrant_client import QdrantClientManager

test_engine = engine
TestingSessionLocal = async_session_factory


@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def cleanup_test_sqlite_file():
    yield
    await test_engine.dispose()
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def in_memory_qdrant() -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
