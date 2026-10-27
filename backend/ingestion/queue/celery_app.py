from celery import Celery
from backend.config import get_settings

settings = get_settings()

celery_app = Celery(
    "financial_rag_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry=False,
    broker_connection_retry_on_startup=False,
    broker_connection_max_retries=0,
    redis_backend_use_ssl=False,
    redis_retry_on_timeout=False,
    result_backend_transport_options={"max_retries": 0, "interval_start": 0, "interval_step": 0, "interval_max": 0},
)
