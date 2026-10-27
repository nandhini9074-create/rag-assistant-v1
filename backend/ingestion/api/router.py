import asyncio
import uuid
from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import get_settings
from backend.constants import DocumentStatus, JobStatus
from backend.database.models import Document, DocumentChunk, IngestionJob
from backend.database.session import get_db
from backend.ingestion.api.schemas import (
    ChunkManifestResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    JobStatusResponse,
)
from backend.ingestion.orchestration.pipeline import IngestionPipelineOrchestrator
from backend.ingestion.parsers.validator import FileValidator
from backend.ingestion.storage.private_store import PrivateStorageManager

import socket
from urllib.parse import urlparse

router = APIRouter(prefix="/documents", tags=["Ingestion"])
settings = get_settings()
storage_manager = PrivateStorageManager()


def is_redis_online(redis_url: str, timeout: float = 0.2) -> bool:
    if not redis_url:
        return False
    try:
        parsed = urlparse(redis_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


async def execute_in_background(document_id: uuid.UUID, job_id: uuid.UUID) -> None:
    orchestrator = IngestionPipelineOrchestrator(storage_manager=storage_manager)
    await orchestrator.execute_pipeline(document_id, job_id)


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Step 1: Upload and register a document for ingestion",
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename",
        )

    file_type_enum = FileValidator.normalize_extension(file.filename)
    document_id = uuid.uuid4()
    job_id = uuid.uuid4()

    try:
        saved_path, file_size = await storage_manager.save_upload_file(
            document_id=document_id,
            file_extension=file_type_enum.value,
            file_stream=file.file,
            max_size_bytes=settings.max_upload_size_bytes,
        )
    except Exception:
        raise

    try:
        FileValidator.validate_content(saved_path, file_type_enum)
    except Exception:
        storage_manager.delete_document_files(document_id)
        raise

    try:
        doc = Document(
            document_id=document_id,
            file_name=file.filename,
            file_type=file_type_enum.value,
            status=DocumentStatus.UPLOADED.value,
            file_size_bytes=file_size,
        )
        job = IngestionJob(
            job_id=job_id,
            document_id=document_id,
            status=JobStatus.QUEUED.value,
            attempt_number=1,
        )
        db.add(doc)
        db.add(job)
        await db.commit()
        await db.refresh(doc)
        await db.refresh(job)
    except Exception as e:
        storage_manager.delete_document_files(document_id)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database registration failed: {str(e)}",
        )

    dispatched_celery = False
    if settings.USE_CELERY and settings.APP_ENV != "test" and is_redis_online(settings.REDIS_URL):
        try:
            from backend.ingestion.queue.tasks import run_ingestion_task
            run_ingestion_task.apply_async(args=[str(document_id), str(job_id)], retry=False)
            dispatched_celery = True
        except Exception:
            pass

    if not dispatched_celery:
        background_tasks.add_task(execute_in_background, document_id, job_id)

    return DocumentUploadResponse(
        document_id=doc.document_id,
        job_id=job.job_id,
        file_name=doc.file_name,
        file_type=doc.file_type,
        status=doc.status,
        job_status=job.status,
        created_at=doc.created_at,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Get document and latest ingestion job status",
)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentStatusResponse:
    doc_stmt = select(Document).where(Document.document_id == document_id)
    doc = (await db.execute(doc_stmt)).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    job_stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(desc(IngestionJob.attempt_number))
    )
    latest_job = (await db.execute(job_stmt)).scalars().first()

    return DocumentStatusResponse(
        document_id=doc.document_id,
        file_name=doc.file_name,
        file_type=doc.file_type,
        status=doc.status,
        total_chunks=doc.total_chunks,
        uploaded_by=doc.uploaded_by,
        file_size_bytes=doc.file_size_bytes,
        failure_reason=doc.failure_reason,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        latest_job=JobStatusResponse.model_validate(latest_job) if latest_job else None,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkManifestResponse],
    summary="Get chunk manifest for a document",
)
async def get_document_chunks(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[ChunkManifestResponse]:
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = (await db.execute(stmt)).scalars().all()
    return [ChunkManifestResponse.model_validate(c) for c in chunks]
