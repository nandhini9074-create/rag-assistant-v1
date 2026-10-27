import asyncio
import uuid
from backend.ingestion.orchestration.pipeline import IngestionPipelineOrchestrator
from backend.ingestion.queue.celery_app import celery_app


@celery_app.task(name="tasks.run_ingestion", bind=True, max_retries=3)
def run_ingestion_task(self, document_id_str: str, job_id_str: str) -> dict[str, str]:
    doc_id = uuid.UUID(document_id_str)
    j_id = uuid.UUID(job_id_str)

    orchestrator = IngestionPipelineOrchestrator()
    try:
        result = asyncio.run(orchestrator.execute_pipeline(doc_id, j_id))
        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
