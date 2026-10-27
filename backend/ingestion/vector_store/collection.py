import uuid
from typing import Any
from qdrant_client import AsyncQdrantClient, models
from backend.config import get_settings


class QdrantCollectionManager:
    """
    Step 9: Dual Named-Vector Collection Setup & Upsert.
    Configures Qdrant with:
    - 'text_english_vector'
    - 'text_arabic_vector'
    """

    @classmethod
    async def ensure_collection_exists(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        english_dim: int,
        arabic_dim: int,
    ) -> None:
        collections = await client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if collection_name not in existing_names:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text_english_vector": models.VectorParams(
                        size=english_dim,
                        distance=models.Distance.COSINE,
                    ),
                    "text_arabic_vector": models.VectorParams(
                        size=arabic_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
            )

            index_fields = ["document_id", "chunk_id", "content_type", "original_language"]
            for field in index_fields:
                try:
                    await client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    @classmethod
    async def upsert_dual_vector_point(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        point_id: uuid.UUID,
        text_english_vector: list[float],
        text_arabic_vector: list[float],
        payload: dict[str, Any],
    ) -> models.UpdateResult:
        point = models.PointStruct(
            id=str(point_id),
            vector={
                "text_english_vector": text_english_vector,
                "text_arabic_vector": text_arabic_vector,
            },
            payload=payload,
        )

        result = await client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True,
        )
        return result
