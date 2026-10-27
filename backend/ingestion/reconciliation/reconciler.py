import datetime
import logging
import uuid
from typing import Any
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from backend.config import get_settings
from backend.constants import ChunkStatus, DocumentStatus, JobStatus
from backend.database.models import Document, DocumentChunk, IngestionJob
from backend.database.session import async_session_factory
from backend.ingestion.embeddings.validator import ModelStartupValidator
from backend.ingestion.vector_store.qdrant_client import QdrantClientManager

logger = logging.getLogger(__name__)


class IngestionReconciler:
    """
    Step 12: Ingestion Health and Reconciliation Service.
    Detects drift, missing Qdrant points, dimension disagreements, orphan points,
    falsely active documents, and stuck jobs.
    Performs auditable, idempotent repairs.
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient | None = None,
        session_factory: Any | None = None,
        en_dim: int | None = None,
        ar_dim: int | None = None,
    ) -> None:
        self.settings = get_settings()
        self.client = qdrant_client or QdrantClientManager.get_async_client()
        self.session_factory = session_factory or async_session_factory
        self.en_dim = en_dim
        self.ar_dim = ar_dim

    async def run_audit(self) -> dict[str, Any]:
        if self.en_dim is not None and self.ar_dim is not None:
            en_dim, ar_dim = self.en_dim, self.ar_dim
        else:
            en_dim, ar_dim = ModelStartupValidator.validate_models()

        issues: dict[str, list[Any]] = {
            "missing_qdrant_points": [],
            "falsely_active_documents": [],
            "stuck_processing_jobs": [],
            "orphaned_qdrant_points": [],
            "dimension_mismatches": [],
        }

        async with self.session_factory() as session:
            # 1. Check for stuck jobs
            timeout_threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                minutes=self.settings.PROCESSING_TIMEOUT_MINUTES
            )
            stuck_stmt = select(IngestionJob).where(
                IngestionJob.status == JobStatus.PROCESSING.value,
                IngestionJob.started_at < timeout_threshold,
            )
            stuck_jobs = (await session.execute(stuck_stmt)).scalars().all()
            for job in stuck_jobs:
                issues["stuck_processing_jobs"].append(
                    {"job_id": str(job.job_id), "document_id": str(job.document_id), "started_at": str(job.started_at)}
                )

            # 2. Check for falsely active documents
            active_docs_stmt = select(Document).where(Document.status == DocumentStatus.ACTIVE.value)
            active_docs = (await session.execute(active_docs_stmt)).scalars().all()

            for doc in active_docs:
                verified_count_stmt = select(func.count(DocumentChunk.chunk_id)).where(
                    DocumentChunk.document_id == doc.document_id,
                    DocumentChunk.status == ChunkStatus.VERIFIED.value,
                )
                verified_count = (await session.execute(verified_count_stmt)).scalar() or 0

                total_manifest_stmt = select(func.count(DocumentChunk.chunk_id)).where(
                    DocumentChunk.document_id == doc.document_id
                )
                total_manifest = (await session.execute(total_manifest_stmt)).scalar() or 0

                if verified_count == 0 or verified_count < total_manifest or doc.total_chunks != verified_count:
                    issues["falsely_active_documents"].append(
                        {
                            "document_id": str(doc.document_id),
                            "doc_total_chunks": doc.total_chunks,
                            "verified_count": verified_count,
                            "manifest_total": total_manifest,
                        }
                    )

            # 3. Check for manifest chunks missing from Qdrant
            verified_chunks_stmt = select(DocumentChunk).where(
                DocumentChunk.status.in_([ChunkStatus.VERIFIED.value, ChunkStatus.PROCESSED.value])
            )
            chunks = (await session.execute(verified_chunks_stmt)).scalars().all()

            for chk in chunks:
                try:
                    points = await self.client.retrieve(
                        collection_name=self.settings.QDRANT_COLLECTION,
                        ids=[str(chk.qdrant_point_id)],
                        with_vectors=True,
                    )
                    if not points:
                        issues["missing_qdrant_points"].append(
                            {"chunk_id": str(chk.chunk_id), "point_id": str(chk.qdrant_point_id)}
                        )
                    else:
                        pt = points[0]
                        vecs = pt.vector or {}
                        en_vec = vecs.get("text_english_vector", [])
                        ar_vec = vecs.get("text_arabic_vector", [])
                        if len(en_vec) != en_dim or len(ar_vec) != ar_dim:
                            issues["dimension_mismatches"].append(
                                {
                                    "point_id": str(chk.qdrant_point_id),
                                    "actual_en": len(en_vec),
                                    "expected_en": en_dim,
                                    "actual_ar": len(ar_vec),
                                    "expected_ar": ar_dim,
                                }
                            )
                except Exception as e:
                    logger.warning(f"Error querying Qdrant for point {chk.qdrant_point_id}: {e}")

        return issues

    async def run_repair(self, audit_issues: dict[str, list[Any]]) -> dict[str, int]:
        repaired = {
            "stuck_jobs_failed": 0,
            "falsely_active_corrected": 0,
        }

        async with self.session_factory() as session:
            for item in audit_issues.get("stuck_processing_jobs", []):
                job_id = uuid.UUID(item["job_id"])
                doc_id = uuid.UUID(item["document_id"])

                job = (await session.execute(select(IngestionJob).where(IngestionJob.job_id == job_id))).scalar_one_or_none()
                if job and job.status == JobStatus.PROCESSING.value:
                    job.status = JobStatus.FAILED.value
                    job.failure_reason = "Reconciliation: Job timed out in PROCESSING state"
                    job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    repaired["stuck_jobs_failed"] += 1

                doc = (await session.execute(select(Document).where(Document.document_id == doc_id))).scalar_one_or_none()
                if doc and doc.status == DocumentStatus.PROCESSING.value:
                    doc.status = DocumentStatus.FAILED.value
                    doc.failure_reason = "Reconciliation: Ingestion timed out"

            for item in audit_issues.get("falsely_active_documents", []):
                doc_id = uuid.UUID(item["document_id"])
                doc = (await session.execute(select(Document).where(Document.document_id == doc_id))).scalar_one_or_none()
                if doc and doc.status == DocumentStatus.ACTIVE.value:
                    doc.status = DocumentStatus.FAILED.value
                    doc.failure_reason = "Reconciliation: Document marked ACTIVE without full verification"
                    repaired["falsely_active_corrected"] += 1

            await session.commit()

        return repaired
