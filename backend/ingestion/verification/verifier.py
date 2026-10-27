import math
import uuid
from typing import Any
from qdrant_client import AsyncQdrantClient


class QdrantStorageVerifier:
    """
    Step 10: Deep Qdrant Storage Verification.
    Validates point existence, both named vectors, dimensions, numeric validity,
    and payload consistency against PostgreSQL manifest.
    """

    @classmethod
    async def verify_point(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        expected_point_id: uuid.UUID,
        expected_chunk_id: uuid.UUID,
        expected_document_id: uuid.UUID,
        expected_en_dim: int,
        expected_ar_dim: int,
        expected_en_model: str,
        expected_ar_model: str,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        points = await client.retrieve(
            collection_name=collection_name,
            ids=[str(expected_point_id)],
            with_vectors=True,
            with_payload=True,
        )

        # Check 1: Point exists
        if not points:
            return False, f"Point '{expected_point_id}' not found in Qdrant collection '{collection_name}'", {}

        point = points[0]
        vectors = point.vector
        payload = point.payload or {}

        # Check 2: Both named vectors present
        if not isinstance(vectors, dict):
            return False, f"Point vectors are not a dictionary of named vectors: {type(vectors)}", payload

        if "text_english_vector" not in vectors:
            return False, "Named vector 'text_english_vector' is missing from stored point", payload

        if "text_arabic_vector" not in vectors:
            return False, "Named vector 'text_arabic_vector' is missing from stored point", payload

        en_vec = vectors["text_english_vector"]
        ar_vec = vectors["text_arabic_vector"]

        # Check 3 & 4: Dimensions match validated models
        if len(en_vec) != expected_en_dim:
            return (
                False,
                f"English vector dimension mismatch: expected {expected_en_dim}, got {len(en_vec)}",
                payload,
            )

        if len(ar_vec) != expected_ar_dim:
            return (
                False,
                f"Arabic vector dimension mismatch: expected {expected_ar_dim}, got {len(ar_vec)}",
                payload,
            )

        # Check 5: Finite values, no NaN/Inf
        for idx, val in enumerate(en_vec):
            if math.isnan(val) or math.isinf(val):
                return False, f"English vector contains NaN or Inf at component {idx}", payload

        for idx, val in enumerate(ar_vec):
            if math.isnan(val) or math.isinf(val):
                return False, f"Arabic vector contains NaN or Inf at component {idx}", payload

        # Check 6: Payload IDs match PostgreSQL
        stored_chunk_id = payload.get("chunk_id")
        stored_doc_id = payload.get("document_id")
        if str(stored_chunk_id) != str(expected_chunk_id):
            return (
                False,
                f"Payload chunk_id mismatch: expected '{expected_chunk_id}', got '{stored_chunk_id}'",
                payload,
            )
        if str(stored_doc_id) != str(expected_document_id):
            return (
                False,
                f"Payload document_id mismatch: expected '{expected_document_id}', got '{stored_doc_id}'",
                payload,
            )

        # Check 7: Text fields and provenance exist
        for key in ["original_text", "text_english", "text_arabic", "original_language"]:
            if not payload.get(key):
                return False, f"Payload is missing required field '{key}'", payload

        # Check 8: Model identifiers match configuration
        emb_models = payload.get("embedding_models", {})
        stored_en_model = emb_models.get("text_english_vector", {}).get("model")
        stored_ar_model = emb_models.get("text_arabic_vector", {}).get("model")

        if stored_en_model != expected_en_model:
            return (
                False,
                f"Stored English model '{stored_en_model}' disagrees with configured '{expected_en_model}'",
                payload,
            )
        if stored_ar_model != expected_ar_model:
            return (
                False,
                f"Stored Arabic model '{stored_ar_model}' disagrees with configured '{expected_ar_model}'",
                payload,
            )

        return True, None, payload
