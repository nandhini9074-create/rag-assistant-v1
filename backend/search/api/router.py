import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.search.orchestration.pipeline import SearchPipelineOrchestrator
from backend.search.api.schemas import (
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger("search-api")

router = APIRouter(prefix="/chat", tags=["Search & Retrieval"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Step 1-11: Full End-to-End Financial Retrieval and Question Answering Pipeline",
    description="Receive question, validate access, detect language, prepare query, encode embedding, retrieve Qdrant candidates, rerank, assemble context, perform deterministic calculation if needed, generate grounded answer, and audit citations/evidence.",
)
async def chat_query(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Step 1: Generates unique tracking request_id (REQ-xxxx).
    Step 2-11: Delegates to SearchPipelineOrchestrator to execute full 11-step retrieval and QA flow.
    """
    request_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"
    orchestrator = SearchPipelineOrchestrator(db=db)
    return await orchestrator.execute_query(
        request_id=request_id,
        request=request,
        user_id=x_user_id,
    )
