import asyncio
import logging
import sys
from contextlib import asynccontextmanager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from backend.config import get_settings
from backend.ingestion.api.router import router as ingestion_router
from backend.ingestion.embeddings.validator import ModelStartupValidator

try:
    from backend.search.api.router import router as search_router
except ImportError:
    search_router = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("financial-rag")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Financial RAG Ingestion Pipeline...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Configured English Model: {settings.ENGLISH_EMBEDDING_MODEL}")
    logger.info(f"Configured Arabic Model: {settings.ARABIC_EMBEDDING_MODEL}")

    try:
        en_dim, ar_dim = ModelStartupValidator.validate_models()
        app.state.english_dim = en_dim
        app.state.arabic_dim = ar_dim
        logger.info(f"Embedding Models validated successfully. Dimensions: English={en_dim}, Arabic={ar_dim}")
    except Exception as e:
        logger.warning(f"Embedding model pre-validation warning: {e}")

    yield

    logger.info("Shutting down Financial RAG Ingestion Pipeline...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial RAG Ingestion API",
        version="0.1.0",
        description="Multilingual Financial RAG Ingestion Flow with Separate English and Arabic Sentence Transformers",
        lifespan=lifespan,
    )

    app.include_router(ingestion_router, prefix=settings.API_V1_PREFIX)
    if search_router is not None:
        app.include_router(search_router, prefix=settings.API_V1_PREFIX)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "environment": settings.APP_ENV,
            "qdrant_collection": settings.QDRANT_COLLECTION,
        }

    @app.get("/", include_in_schema=False)
    @app.get("/docs.", include_in_schema=False)
    async def redirect_to_docs():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    return app


app = create_app()
