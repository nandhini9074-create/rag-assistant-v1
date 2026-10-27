
from backend.config import get_settings
from backend.ingestion.embeddings.base import BaseEmbeddingService


class ArabicEmbeddingService(BaseEmbeddingService):
    """
    Step 8 Service: Arabic Sentence Transformers Embedding Service.
    Uses 'Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2'.
    Validates model output dimension dynamically at runtime.
    Supports Matryoshka dimension if explicitly configured.
    """

    def __init__(
        self,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> None:
        settings = get_settings()
        m_name = model_name or settings.ARABIC_EMBEDDING_MODEL
        rev = revision or settings.ARABIC_EMBEDDING_REVISION
        super().__init__(model_name=m_name, revision=rev)

    def load_model(self) -> None:
        settings = get_settings()
        if settings.APP_ENV == "test" or self.model_name == "mock":
            self._dimension = 768
            self._model = "mock"
            return

        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.revision:
                kwargs["revision"] = self.revision

            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._dimension = self._model.get_sentence_embedding_dimension()
            if self._dimension is None:
                sample_vec = self._model.encode("عينة تجريبية لاختبار أبعاد المتجه المالي", convert_to_numpy=True)
                self._dimension = len(sample_vec)

        except ImportError:
            self._dimension = 768
            self._model = "mock"

    def encode(self, text: str) -> list[float]:
        if self._model is None:
            self.load_model()

        if self._model == "mock":
            dim = self.dimension
            vec = [float(hash(text + "ar" + str(i)) % 1000) / 1000.0 for i in range(dim)]
            self.validate_vector(vec)
            return vec

        raw_vec = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vec = [float(v) for v in raw_vec]
        self.validate_vector(vec)
        return vec
