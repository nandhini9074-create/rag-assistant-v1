import logging
from backend.ingestion.embeddings.arabic_service import ArabicEmbeddingService
from backend.ingestion.embeddings.english_service import EnglishEmbeddingService

logger = logging.getLogger(__name__)


class ModelStartupValidator:
    """
    Validates English and Arabic Sentence Transformers models at application startup.
    Determines and verifies exact output dimensions without hardcoding.
    """

    @classmethod
    def validate_models(
        cls,
        english_service: EnglishEmbeddingService | None = None,
        arabic_service: ArabicEmbeddingService | None = None,
    ) -> tuple[int, int]:
        en_svc = english_service or EnglishEmbeddingService()
        ar_svc = arabic_service or ArabicEmbeddingService()

        logger.info(f"Loading and validating English model: {en_svc.model_name}")
        en_svc.load_model()
        en_dim = en_svc.dimension

        logger.info(f"Loading and validating Arabic model: {ar_svc.model_name}")
        ar_svc.load_model()
        ar_dim = ar_svc.dimension

        logger.info(f"Successfully validated model dimensions: English={en_dim}, Arabic={ar_dim}")
        return en_dim, ar_dim
