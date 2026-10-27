from backend.ingestion.queue.celery_app import celery_app
from backend.ingestion.queue.tasks import run_ingestion_task

__all__ = ["celery_app", "run_ingestion_task"]
