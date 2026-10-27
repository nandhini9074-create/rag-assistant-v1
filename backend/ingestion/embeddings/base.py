import math
from abc import ABC, abstractmethod


class BaseEmbeddingService(ABC):
    """
    Abstract Base Class for Sentence Transformers embedding services.
    Enforces separate models, independent revisions, and dimension validation.
    """

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        self.model_name = model_name
        self.revision = revision
        self._model = None
        self._dimension: int | None = None

    @abstractmethod
    def load_model(self) -> None:
        pass

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(f"Model '{self.model_name}' has not been loaded and validated yet")
        return self._dimension

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        pass

    def validate_vector(self, vector: list[float]) -> None:
        if not vector:
            raise ValueError(f"Model '{self.model_name}' produced an empty vector")

        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch for '{self.model_name}': expected {self.dimension}, got {len(vector)}"
            )

        for idx, val in enumerate(vector):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Vector contains invalid numeric value (NaN or Inf) at index {idx}")
