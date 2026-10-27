import uuid
import pytest
from qdrant_client import AsyncQdrantClient
from backend.ingestion.vector_store.collection import QdrantCollectionManager
from backend.ingestion.verification.verifier import QdrantStorageVerifier


@pytest.mark.asyncio
async def test_qdrant_dual_vector_storage_and_verification(in_memory_qdrant: AsyncQdrantClient):
    collection_name = "test_financial_documents"
    en_dim = 384
    ar_dim = 768
    en_model = "sentence-transformers/all-MiniLM-L6-v2"
    ar_model = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"

    # Step 9: Create collection with dual named vectors
    await QdrantCollectionManager.ensure_collection_exists(
        client=in_memory_qdrant,
        collection_name=collection_name,
        english_dim=en_dim,
        arabic_dim=ar_dim,
    )

    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    point_id = uuid.uuid4()

    en_vec = [0.1] * en_dim
    ar_vec = [0.2] * ar_dim

    payload = {
        "chunk_id": str(chunk_id),
        "document_id": str(doc_id),
        "element_ids": [str(uuid.uuid4())],
        "chunk_index": 0,
        "content_type": "financial_table",
        "original_language": "en",
        "original_text": "Revenue: AED 50000",
        "text_english": "Revenue: AED 50000",
        "text_arabic": "الإيرادات: 50000 درهم إماراتي",
        "translation_status": "COMPLETED",
        "source": {"page": 1},
        "embedding_models": {
            "text_english_vector": {"model": en_model, "revision": None, "dimensions": en_dim},
            "text_arabic_vector": {"model": ar_model, "revision": None, "dimensions": ar_dim},
        },
    }

    # Upsert single point with dual named vectors
    await QdrantCollectionManager.upsert_dual_vector_point(
        client=in_memory_qdrant,
        collection_name=collection_name,
        point_id=point_id,
        text_english_vector=en_vec,
        text_arabic_vector=ar_vec,
        payload=payload,
    )

    # Step 10: Run deep verification
    is_verified, reason, verified_payload = await QdrantStorageVerifier.verify_point(
        client=in_memory_qdrant,
        collection_name=collection_name,
        expected_point_id=point_id,
        expected_chunk_id=chunk_id,
        expected_document_id=doc_id,
        expected_en_dim=en_dim,
        expected_ar_dim=ar_dim,
        expected_en_model=en_model,
        expected_ar_model=ar_model,
    )

    assert is_verified is True
    assert reason is None
    assert verified_payload["original_language"] == "en"


@pytest.mark.asyncio
async def test_qdrant_verification_fails_on_dimension_mismatch(in_memory_qdrant: AsyncQdrantClient):
    collection_name = "test_financial_documents_dim_err"
    en_dim = 384
    ar_dim = 768

    await QdrantCollectionManager.ensure_collection_exists(
        client=in_memory_qdrant,
        collection_name=collection_name,
        english_dim=en_dim,
        arabic_dim=ar_dim,
    )

    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    point_id = uuid.uuid4()

    payload = {
        "chunk_id": str(chunk_id),
        "document_id": str(doc_id),
        "original_language": "en",
        "original_text": "Test",
        "text_english": "Test",
        "text_arabic": "اختبار",
        "embedding_models": {
            "text_english_vector": {"model": "model-en"},
            "text_arabic_vector": {"model": "model-ar"},
        },
    }

    await QdrantCollectionManager.upsert_dual_vector_point(
        client=in_memory_qdrant,
        collection_name=collection_name,
        point_id=point_id,
        text_english_vector=[0.1] * en_dim,
        text_arabic_vector=[0.2] * ar_dim,
        payload=payload,
    )

    # Expect dimension 512 instead of 384
    is_verified, reason, _ = await QdrantStorageVerifier.verify_point(
        client=in_memory_qdrant,
        collection_name=collection_name,
        expected_point_id=point_id,
        expected_chunk_id=chunk_id,
        expected_document_id=doc_id,
        expected_en_dim=512,  # Intentionally wrong
        expected_ar_dim=ar_dim,
        expected_en_model="model-en",
        expected_ar_model="model-ar",
    )

    assert is_verified is False
    assert "English vector dimension mismatch" in reason
