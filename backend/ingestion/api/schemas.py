import uuid
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    job_id: uuid.UUID
    file_name: str
    file_type: str
    status: str
    job_status: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID
    document_id: uuid.UUID
    status: str
    current_step: Optional[str] = None
    parser: Optional[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    verified_chunks: int = 0
    attempt_number: int = 1
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_step: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    file_name: str
    file_type: str
    status: str
    total_chunks: int = 0
    uploaded_by: Optional[str] = None
    file_size_bytes: Optional[int] = None
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    latest_job: Optional[JobStatusResponse] = None


class ChunkManifestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    job_id: uuid.UUID
    chunk_index: int
    content_type: str
    original_language: Optional[str] = None
    translation_status: str
    qdrant_point_id: uuid.UUID
    source_metadata: dict[str, Any]
    status: str
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
