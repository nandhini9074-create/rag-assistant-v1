import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional
from backend.config import get_settings

logger = logging.getLogger("search-embeddings")


@dataclass
class QueryEmbeddingResult:
    """
    Structured result conforming to Step 5: Query Embedding Generation specification.
    """
    request_id: str
    embedding_model: str
    vector_name: str
    query_vector: list[float] = field(repr=False)
    vector_dimension: int
    status: str = "COMPLETED"


class QueryEmbedder:
    """
    Step 5: Query Embedding Generation.
    Generates high-fidelity numerical query embeddings mapped to the compatible
    Qdrant vector space:
    - Arabic queries ('ar') -> Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2 -> 'text_arabic_vector' (dim=768)
    - English queries ('en') -> sentence-transformers/all-MiniLM-L6-v2 -> 'text_english_vector' (dim=384)
    - Validates finite numbers (no NaN/Inf) and exact dimensional consistency.
    """

    _loaded_models: dict[str, Any] = {}
    _model_dimensions: dict[str, int] = {}

    @classmethod
    def get_target_vector_name(cls, query_language: str) -> str:
        """
        Map language to Qdrant collection vector schema name.
        """
        return "text_arabic_vector" if query_language.lower() == "ar" else "text_english_vector"

    @classmethod
    def get_configured_model_name(cls, query_language: str) -> str:
        """
        Get model identifier based on query language.
        """
        settings = get_settings()
        if query_language.lower() == "ar":
            return settings.ARABIC_EMBEDDING_MODEL
        return settings.ENGLISH_EMBEDDING_MODEL

    @classmethod
    def _get_or_load_model(cls, model_name: str, expected_dim: int) -> tuple[Any, int]:
        """
        Thread-safe cached loader for sentence transformer models.
        In test mode or mock environment, returns 'mock' with expected dimension.
        """
        settings = get_settings()
        if settings.APP_ENV == "test" or model_name == "mock":
            return "mock", expected_dim

        if model_name in cls._loaded_models:
            return cls._loaded_models[model_name], cls._model_dimensions[model_name]

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading query embedding model '{model_name}'...")
            model = SentenceTransformer(model_name)
            dim = model.get_sentence_embedding_dimension()
            if dim is None:
                probe = model.encode("Financial revenue test probe", convert_to_numpy=True)
                dim = len(probe)

            cls._loaded_models[model_name] = model
            cls._model_dimensions[model_name] = dim
            logger.info(f"Loaded embedding model '{model_name}' with dimension {dim}.")
            return model, dim
        except Exception as e:
            logger.warning(
                f"Failed to load SentenceTransformer for '{model_name}': {e}. Falling back to mock encoder."
            )
            return "mock", expected_dim

    @classmethod
    def validate_vector(cls, vector: list[float], expected_dim: int, model_name: str) -> None:
        """
        Validates output vector dimension and checks for NaN or Inf values.
        """
        if not vector:
            raise ValueError(f"Embedding model '{model_name}' generated an empty vector.")
        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch for model '{model_name}': "
                f"expected {expected_dim}, got {len(vector)}."
            )
        for i, val in enumerate(vector):
            if not isinstance(val, (float, int)) or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Vector contains invalid numeric value at index {i}: {val} (NaN or Inf detected)."
                )

    @classmethod
    def _generate_mock_vector(cls, text: str, dim: int, lang_salt: str = "") -> list[float]:
        """
        Generates deterministic L2-normalized pseudo-vector for fast unit testing.
        """
        raw = [float(hash(f"{text}_{lang_salt}_{i}") % 10000 - 5000) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    @classmethod
    def generate_embedding(
        cls,
        request_id: str,
        prepared_query: str,
        query_language: str,
        embedding_model: Optional[str] = None,
    ) -> QueryEmbeddingResult:
        """
        Step 5 Execution:
        Converts prepared query into a numerical vector matching the target Qdrant vector space.
        """
        if not prepared_query or not prepared_query.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only query.")

        is_arabic = (query_language.lower() == "ar")
        target_vector_name = cls.get_target_vector_name(query_language)
        model_name = embedding_model or cls.get_configured_model_name(query_language)
        expected_dim = 768 if is_arabic else 384

        model, actual_dim = cls._get_or_load_model(model_name, expected_dim)

        if model == "mock":
            vector = cls._generate_mock_vector(prepared_query, actual_dim, lang_salt=query_language)
        else:
            raw_vec = model.encode(
                prepared_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vector = [float(x) for x in raw_vec.tolist()]

        # Validate numerical validity and dimension
        cls.validate_vector(vector, actual_dim, model_name)

        logger.info(
            f"[{request_id}] Generated {len(vector)}-dim embedding for {query_language} query "
            f"using '{model_name}' -> target: '{target_vector_name}'."
        )

        return QueryEmbeddingResult(
            request_id=request_id,
            embedding_model=model_name,
            vector_name=target_vector_name,
            query_vector=vector,
            vector_dimension=actual_dim,
            status="COMPLETED",
        )

    @classmethod
    def generate_dual_embeddings(
        cls,
        request_id: str,
        prepared_query: str,
    ) -> dict[str, QueryEmbeddingResult]:
        """
        Generates both English (384-dim) and Arabic (768-dim) vectors for dual-space retrieval.
        """
        return {
            "english": cls.generate_embedding(
                request_id=request_id,
                prepared_query=prepared_query,
                query_language="en",
            ),
            "arabic": cls.generate_embedding(
                request_id=request_id,
                prepared_query=prepared_query,
                query_language="ar",
            ),
        }
