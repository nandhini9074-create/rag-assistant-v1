import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from qdrant_client import AsyncQdrantClient, models
from backend.config import get_settings
from backend.search.api.schemas import RetrievedCandidate

logger = logging.getLogger("search-retrieval")


@dataclass
class RetrievalResult:
    """
    Step 6: Qdrant Vector Retrieval Output Schema.
    """
    request_id: str
    retrieval_status: str
    results: list[RetrievedCandidate] = field(default_factory=list)
    result_count: int = 0
    vector_name: str = ""
    collection: str = ""


class QdrantRetriever:
    """
    Step 6: Qdrant Vector Retrieval Service.
    Finds the most semantically relevant document chunks using vector similarity
    while strictly restricting results to authorized documents.
    """

    def __init__(
        self,
        client: Optional[AsyncQdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        if client is not None:
            self.client = client
        else:
            from backend.ingestion.vector_store.qdrant_client import QdrantClientManager
            self.client = QdrantClientManager.get_async_client()

    async def retrieve(
        self,
        request_id: str,
        query_vector: list[float],
        vector_name: str,
        authorized_document_ids: Sequence[str | uuid.UUID],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Execute Step 6 vector retrieval against Qdrant collection:
        - Validates that authorized document scope is non-empty.
        - Constructs exact document_id filter.
        - Queries Qdrant using the specified named vector ('text_arabic_vector' or 'text_english_vector').
        - Returns structured candidates with scores, texts, and source metadata.
        """
        # Security Guard: Empty authorized scope must NEVER result in an open, unrestricted vector search
        if not authorized_document_ids:
            logger.warning(
                f"[{request_id}] QdrantRetriever received empty authorized_document_ids. "
                "Returning zero candidates to enforce security boundary."
            )
            return RetrievalResult(
                request_id=request_id,
                retrieval_status="COMPLETED",
                results=[],
                result_count=0,
                vector_name=vector_name,
                collection=self.collection_name,
            )

        str_doc_ids = [str(did) for did in authorized_document_ids]

        # Construct mandatory Qdrant payload filter for authorized documents
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=str_doc_ids),
                )
            ]
        )

        logger.info(
            f"[{request_id}] Querying Qdrant '{self.collection_name}' (vector='{vector_name}', "
            f"top_k={top_k}) scoped to {len(str_doc_ids)} document(s)..."
        )

        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.error(
                f"[{request_id}] Qdrant query_points failed on collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return RetrievalResult(
                request_id=request_id,
                retrieval_status="FAILED",
                results=[],
                result_count=0,
                vector_name=vector_name,
                collection=self.collection_name,
            )

        candidates: list[RetrievedCandidate] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point.id)
            doc_id = str(payload.get("document_id") or "")
            sim_score = float(point.score) if point.score is not None else 0.0

            source_dict = payload.get("source") or {}
            if not isinstance(source_dict, dict):
                source_dict = {"raw_source": str(source_dict)}

            cand = RetrievedCandidate(
                chunk_id=chunk_id,
                document_id=doc_id,
                retrieval_score=round(sim_score, 4),
                relevance_rank=len(candidates) + 1,
                relevance_score=round(sim_score, 4),
                text_english=str(payload.get("text_english") or payload.get("original_text") or ""),
                text_arabic=str(payload.get("text_arabic") or payload.get("original_text") or ""),
                original_text=str(payload.get("original_text") or ""),
                original_language=str(payload.get("original_language") or "en"),
                source=source_dict,
            )
            candidates.append(cand)

        logger.info(
            f"[{request_id}] Successfully retrieved {len(candidates)} candidate chunk(s) from Qdrant."
        )

        return RetrievalResult(
            request_id=request_id,
            retrieval_status="COMPLETED",
            results=candidates,
            result_count=len(candidates),
            vector_name=vector_name,
            collection=self.collection_name,
        )
