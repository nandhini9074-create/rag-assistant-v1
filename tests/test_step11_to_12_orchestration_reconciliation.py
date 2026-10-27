import datetime
import io
import uuid
from pathlib import Path
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.constants import ChunkStatus, DocumentStatus, JobStatus
from backend.database.models import Document, DocumentChunk, IngestionJob
from backend.ingestion.orchestration.pipeline import IngestionPipelineOrchestrator
from backend.ingestion.reconciliation.reconciler import IngestionReconciler
from backend.ingestion.storage.private_store import PrivateStorageManager


from qdrant_client import AsyncQdrantClient
from tests.conftest import TestingSessionLocal


@pytest.mark.asyncio
async def test_end_to_end_orchestration_activates_document(
    db_session: AsyncSession,
    in_memory_qdrant: AsyncQdrantClient,
    tmp_path: Path,
):
    storage = PrivateStorageManager(root_dir=tmp_path / "private_storage")
    doc_id = uuid.uuid4()
    job_id = uuid.uuid4()

    # Create test CSV file in private storage
    csv_content = b"Month,Revenue,Currency\nApril,50000,AED\nMay,60000,AED\n"
    await storage.save_upload_file(
        document_id=doc_id,
        file_extension="csv",
        file_stream=io.BytesIO(csv_content),
        max_size_bytes=10 * 1024 * 1024,
    )

    doc = Document(
        document_id=doc_id,
        file_name="financials.csv",
        file_type="csv",
        status=DocumentStatus.UPLOADED.value,
    )
    job = IngestionJob(
        job_id=job_id,
        document_id=doc_id,
        status=JobStatus.QUEUED.value,
        attempt_number=1,
    )
    db_session.add(doc)
    db_session.add(job)
    await db_session.commit()

    # Run pipeline orchestrator
    orchestrator = IngestionPipelineOrchestrator(
        storage_manager=storage,
        session_factory=TestingSessionLocal,
        qdrant_client=in_memory_qdrant,
    )
    result = await orchestrator.execute_pipeline(doc_id, job_id)

    assert result["document_status"] == "ACTIVE"
    assert result["job_status"] == "COMPLETED"

    # Query updated DB state
    await db_session.refresh(doc)
    await db_session.refresh(job)

    assert doc.status == DocumentStatus.ACTIVE.value
    assert doc.total_chunks > 0
    assert job.status == JobStatus.COMPLETED.value
    assert job.verified_chunks == doc.total_chunks


@pytest.mark.asyncio
async def test_reconciliation_detects_falsely_active_document(
    db_session: AsyncSession,
    in_memory_qdrant: AsyncQdrantClient,
):
    doc_id = uuid.uuid4()
    # Document marked ACTIVE but with total_chunks = 5 and 0 verified chunks in manifest
    doc = Document(
        document_id=doc_id,
        file_name="fake_active.pdf",
        file_type="pdf",
        status=DocumentStatus.ACTIVE.value,
        total_chunks=5,
    )
    db_session.add(doc)
    await db_session.commit()

    reconciler = IngestionReconciler(
        qdrant_client=in_memory_qdrant,
        session_factory=TestingSessionLocal,
        en_dim=384,
        ar_dim=768,
    )
    issues = await reconciler.run_audit()

    falsely_active = [item for item in issues["falsely_active_documents"] if item["document_id"] == str(doc_id)]
    assert len(falsely_active) == 1

    # Run repair: should demote falsely active document to FAILED
    repair_summary = await reconciler.run_repair(issues)
    assert repair_summary["falsely_active_corrected"] >= 1

    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED.value
    assert "without full verification" in doc.failure_reason
