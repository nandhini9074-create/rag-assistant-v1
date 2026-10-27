import logging
import uuid
from typing import Optional
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession
from backend.search.access_control.resolver import DocumentAccessResolver
from backend.search.language.detector import QueryLanguageDetector
from backend.search.preprocessing.query_cleaner import QueryPreprocessor
from backend.search.embeddings.query_embedder import QueryEmbedder
from backend.search.retrieval.qdrant_retriever import QdrantRetriever
from backend.search.reranking.reranker import ResultReranker
from backend.search.context.builder import ContextBuilder
from backend.search.calculation.engine import FinancialCalculationEngine
from backend.search.generation.answer_generator import AnswerGenerator
from backend.search.validation.citation_verifier import CitationAndEvidenceVerifier
from backend.search.api.schemas import (
    AnswerMetadata,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger("search-orchestrator")


class SearchPipelineOrchestrator:
    """
    Part 11: Search Pipeline Orchestrator.
    Coordinates the complete 11-step Arabic–English financial retrieval and question answering workflow:
    - Step 1: Request initialization and tracking (REQ-xxxx).
    - Step 2: Request validation, document resolution & tenant authorization.
    - Step 3: Query language detection (Arabic vs English).
    - Step 4: Query preparation & financial entity extraction.
    - Step 5: Query embedding generation matching target Qdrant vector space.
    - Step 6: Qdrant vector retrieval with authorized document filters.
    - Step 7: Result reranking, token matching, and score threshold filtering.
    - Step 8: Token-bounded context assembly with language-aware text representation.
    - Part 8: Deterministic financial calculation engine for aggregations and differences.
    - Step 9: Evidence-grounded answer generation with explicit citations.
    - Step 10: Citation existence verification and numerical claim audit.
    - Step 11: Final typed response delivery conforming to API schema.
    """

    def __init__(
        self,
        db: AsyncSession,
        qdrant_client: Optional[AsyncQdrantClient] = None,
    ) -> None:
        self.db = db
        self.qdrant_client = qdrant_client

    async def execute_query(
        self,
        request_id: str,
        request: QueryRequest,
        user_id: Optional[str] = None,
    ) -> QueryResponse:
        """
        Execute Steps 1 to 11 synchronously and return the audited QueryResponse.
        """
        conversation_id = request.conversation_id or f"CONV-{uuid.uuid4().hex[:6].upper()}"
        logger.info(
            f"[{request_id}] Pipeline initiated for query: '{request.query[:80]}...' "
            f"| docs: {request.document_names} | user: {user_id}"
        )

        # Step 2: Access Control & Resolution
        resolver = DocumentAccessResolver(self.db)
        val_result = await resolver.validate_and_resolve(
            request_id=request_id,
            query=request.query,
            document_names=request.document_names,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        # Step 3: Query Language Detection
        lang_result = QueryLanguageDetector.detect_language(
            request_id=request_id,
            query=val_result.query,
        )

        # Step 4: Query Preparation & Token Extraction
        prep_result = QueryPreprocessor.prepare_query(
            request_id=request_id,
            query=val_result.query,
            query_language=lang_result.query_language,
        )

        # Guard: Empty authorized scope check
        if not val_result.authorized_document_ids:
            logger.info(f"[{request_id}] Empty authorized document scope; returning INSUFFICIENT_EVIDENCE.")
            insufficient_msg = (
                "لم يتم العثور على أي مستندات مالية نشطة مصرح بها للإجابة على سؤالك."
                if lang_result.response_language == "ar"
                else "No active financial documents are available in the authorized scope to answer your question."
            )
            return QueryResponse(
                request_id=request_id,
                conversation_id=conversation_id,
                status="INSUFFICIENT_EVIDENCE",
                answer=insufficient_msg,
                response_language=lang_result.response_language,
                citations=[],
                answer_metadata=AnswerMetadata(
                    citation_validation="PASSED",
                    evidence_validation="INSUFFICIENT_EVIDENCE",
                    calculation_performed=False,
                    unsupported_claims=["No active authorized documents found in database."],
                ),
            )

        # Step 5: Query Embedding Generation
        emb_result = QueryEmbedder.generate_embedding(
            request_id=request_id,
            prepared_query=prep_result.prepared_query,
            query_language=prep_result.query_language,
        )

        # Step 6: Qdrant Vector Retrieval
        retriever = QdrantRetriever(client=self.qdrant_client)
        ret_result = await retriever.retrieve(
            request_id=request_id,
            query_vector=emb_result.query_vector,
            vector_name=emb_result.vector_name,
            authorized_document_ids=val_result.authorized_document_ids,
            top_k=request.top_k,
        )

        # Step 7: Result Reranking & Relevance Filtering
        rerank_result = ResultReranker.rerank_and_filter(
            request_id=request_id,
            query=prep_result.prepared_query,
            candidates=ret_result.results,
            financial_tokens=prep_result.financial_tokens,
            min_score_threshold=0.35,
            max_candidates=request.top_k,
        )

        # Step 8: Context Building
        ctx_result = ContextBuilder.build_context(
            request_id=request_id,
            query=prep_result.prepared_query,
            response_language=lang_result.response_language,
            candidates=rerank_result.results,
            max_context_tokens=4000,
        )

        # Part 8: Deterministic Financial Calculation Engine
        calc_result = FinancialCalculationEngine.compute(
            request_id=request_id,
            query=prep_result.prepared_query,
            context_chunks=ctx_result.context,
        )

        # If deterministic calculation was performed, append verified proof to context
        formatted_context = ctx_result.formatted_context_str
        if calc_result.performed:
            proof_block = (
                f"\n[Verified Deterministic Calculation Proof]\n"
                f"- Operation: {calc_result.operation}\n"
                f"- Formula: {calc_result.formula_str}\n"
                f"- Result: {calc_result.result_value} {calc_result.currency}\n"
            )
            formatted_context += proof_block

        # Step 9: Answer Generation
        generator = AnswerGenerator()
        draft = await generator.generate_draft_answer(
            request_id=request_id,
            original_query=prep_result.prepared_query,
            response_language=lang_result.response_language,
            formatted_context_str=formatted_context,
            context_chunks=ctx_result.context,
        )

        # Step 10: Evidence & Citation Validation
        allowed_figures = [str(calc_result.result_value)] if calc_result.performed and calc_result.result_value is not None else None
        ver_result = CitationAndEvidenceVerifier.validate_answer_and_citations(
            request_id=request_id,
            answer=draft.answer,
            context_chunks=ctx_result.context,
            citations=draft.citations,
            allowed_verified_figures=allowed_figures,
        )

        # Step 11: Final API Response Construction
        final_status = "COMPLETED"
        if draft.generation_status == "INSUFFICIENT_EVIDENCE":
            final_status = "INSUFFICIENT_EVIDENCE"
        elif not ver_result.ready_to_return:
            final_status = "FAILED"

        logger.info(
            f"[{request_id}] Pipeline completed successfully: status='{final_status}', "
            f"citations={len(ver_result.verified_citations)}, calc={calc_result.performed}"
        )

        return QueryResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            status=final_status,
            answer=ver_result.validated_answer,
            response_language=lang_result.response_language,
            citations=ver_result.verified_citations,
            answer_metadata=AnswerMetadata(
                citation_validation=ver_result.citation_validation,
                evidence_validation=ver_result.evidence_validation,
                calculation_performed=calc_result.performed,
                unsupported_claims=ver_result.unsupported_claims,
            ),
        )
