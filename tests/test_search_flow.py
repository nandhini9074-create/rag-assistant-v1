import uuid
import pytest
from httpx import AsyncClient
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession
from backend.constants import DocumentStatus
from backend.database.models import Document


@pytest.mark.asyncio
async def test_search_part1_and_part2_arabic_query_with_active_document(
    async_client: AsyncClient,
    db_session: AsyncSession,
):
    # Insert an ACTIVE document in database
    doc_id = uuid.uuid4()
    doc = Document(
        document_id=doc_id,
        file_name="April_Financial_Report.pdf",
        file_type="pdf",
        status=DocumentStatus.ACTIVE.value,
        total_chunks=3,
    )
    db_session.add(doc)
    await db_session.commit()

    payload = {
        "query": "ما هو إجمالي الإيرادات لشهر أبريل؟",
        "document_names": ["April_Financial_Report.pdf"],
        "conversation_id": "CONV-1001",
    }
    response = await async_client.post("/api/v1/chat/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"].startswith("REQ-")
    assert data["conversation_id"] == "CONV-1001"
    assert data["status"] in ("COMPLETED", "INSUFFICIENT_EVIDENCE")
    assert data["response_language"] == "ar"
    assert "answer" in data
    assert "citations" in data
    assert "answer_metadata" in data


@pytest.mark.asyncio
async def test_search_part2_unknown_document_returns_404(async_client: AsyncClient):
    payload = {
        "query": "What is the revenue?",
        "document_names": ["Ghost_Document_DoesNotExist.pdf"],
    }
    response = await async_client.post("/api/v1/chat/query", json=payload)
    assert response.status_code == 404
    assert "not found in the document registry" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_part2_inactive_document_returns_400(
    async_client: AsyncClient,
    db_session: AsyncSession,
):
    # Insert a FAILED / non-active document
    doc = Document(
        document_id=uuid.uuid4(),
        file_name="failed_draft.pdf",
        file_type="pdf",
        status=DocumentStatus.FAILED.value,
    )
    db_session.add(doc)
    await db_session.commit()

    payload = {
        "query": "What are the numbers?",
        "document_names": ["failed_draft.pdf"],
    }
    response = await async_client.post("/api/v1/chat/query", json=payload)
    assert response.status_code == 400
    assert "Only ACTIVE documents can be searched" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_part2_unauthorized_document_returns_403(
    async_client: AsyncClient,
    db_session: AsyncSession,
):
    # Insert document belonging to Alice
    doc = Document(
        document_id=uuid.uuid4(),
        file_name="alice_confidential.pdf",
        file_type="pdf",
        status=DocumentStatus.ACTIVE.value,
        uploaded_by="user_alice",
    )
    db_session.add(doc)
    await db_session.commit()

    # User Bob attempts to access it
    payload = {
        "query": "Show me secret numbers",
        "document_names": ["alice_confidential.pdf"],
    }
    response = await async_client.post(
        "/api/v1/chat/query",
        json=payload,
        headers={"X-User-Id": "user_bob"},
    )
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_part2_unscoped_query_with_no_active_documents(async_client: AsyncClient):
    # Empty DB has no active documents, must return INSUFFICIENT_EVIDENCE
    payload = {
        "query": "What was the total revenue in Q1?",
        "document_names": None,
    }
    response = await async_client.post("/api/v1/chat/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert data["answer_metadata"]["evidence_validation"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_search_part1_empty_query_rejected(async_client: AsyncClient):
    payload = {
        "query": "",
    }
    response = await async_client.post("/api/v1/chat/query", json=payload)
    assert response.status_code == 422


def test_search_part3_language_detector_arabic():
    from backend.search.language.detector import QueryLanguageDetector
    res = QueryLanguageDetector.detect_language("REQ-1", "ما هو إجمالي الإيرادات لشهر أبريل؟")
    assert res.query_language == "ar"
    assert res.response_language == "ar"
    assert res.confidence >= 0.95
    assert res.status == "COMPLETED"


def test_search_part3_language_detector_english():
    from backend.search.language.detector import QueryLanguageDetector
    res = QueryLanguageDetector.detect_language("REQ-2", "What was the total operating income in 2026?")
    assert res.query_language == "en"
    assert res.response_language == "en"
    assert res.confidence >= 0.95


def test_search_part3_language_detector_mixed_and_symbol_fallback():
    from backend.search.language.detector import QueryLanguageDetector
    # Arabic dominant mixed query
    res_ar_mixed = QueryLanguageDetector.detect_language("REQ-3", "ما هو الـ Revenue لعام 2026 في دبي؟")
    assert res_ar_mixed.query_language == "ar"

    # Numbers and symbols only fallback
    res_symbols = QueryLanguageDetector.detect_language("REQ-4", "50000 + 12000?")
    assert res_symbols.query_language == "en"
    assert res_symbols.confidence == 0.50


def test_search_part4_query_preprocessor_whitespace_and_tokens():
    from backend.search.preprocessing.query_cleaner import QueryPreprocessor
    messy_query = "   What   was   the   Q1   2026   revenue   in   AED  ?   (+15.5%)   \u200b\ufeff "
    res = QueryPreprocessor.prepare_query("REQ-5", messy_query, "en")
    assert res.request_id == "REQ-5"
    assert res.prepared_query == "What was the Q1 2026 revenue in AED ? (+15.5%)"
    assert "Q1" in res.financial_tokens
    assert "2026" in res.financial_tokens
    assert "AED" in res.financial_tokens
    assert "+15.5%" in res.financial_tokens
    assert res.status == "COMPLETED"


def test_search_part4_query_preprocessor_arabic_preservation():
    from backend.search.preprocessing.query_cleaner import QueryPreprocessor
    arabic_query = "  ما   هو إجمالي الإيرادات لشهر أبريل في دبي بمبلغ 50,000 AED؟  "
    res = QueryPreprocessor.prepare_query("REQ-6", arabic_query, "ar")
    assert res.prepared_query == "ما هو إجمالي الإيرادات لشهر أبريل في دبي بمبلغ 50,000 AED؟"
    assert "50,000" in res.financial_tokens
    assert "AED" in res.financial_tokens


def test_search_part4_query_embedder_arabic():
    from backend.search.embeddings.query_embedder import QueryEmbedder
    res = QueryEmbedder.generate_embedding(
        request_id="REQ-7",
        prepared_query="ما هو إجمالي الإيرادات لشهر أبريل؟",
        query_language="ar",
    )
    assert res.request_id == "REQ-7"
    assert res.vector_name == "text_arabic_vector"
    assert res.vector_dimension == 768
    assert len(res.query_vector) == 768
    assert res.status == "COMPLETED"
    assert all(isinstance(x, float) for x in res.query_vector)


def test_search_part4_query_embedder_english():
    from backend.search.embeddings.query_embedder import QueryEmbedder
    res = QueryEmbedder.generate_embedding(
        request_id="REQ-8",
        prepared_query="What was the total operating income in Q1 2026?",
        query_language="en",
    )
    assert res.request_id == "REQ-8"
    assert res.vector_name == "text_english_vector"
    assert res.vector_dimension == 384
    assert len(res.query_vector) == 384
    assert res.status == "COMPLETED"
    assert all(isinstance(x, float) for x in res.query_vector)


def test_search_part4_query_embedder_dual_vectors():
    from backend.search.embeddings.query_embedder import QueryEmbedder
    dual = QueryEmbedder.generate_dual_embeddings(
        request_id="REQ-9",
        prepared_query="Operating revenue comparison",
    )
    assert "english" in dual
    assert "arabic" in dual
    assert dual["english"].vector_name == "text_english_vector"
    assert dual["english"].vector_dimension == 384
    assert dual["arabic"].vector_name == "text_arabic_vector"
    assert dual["arabic"].vector_dimension == 768


def test_search_part4_query_embedder_validation():
    import pytest
    from backend.search.embeddings.query_embedder import QueryEmbedder
    # Empty query raises ValueError
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        QueryEmbedder.generate_embedding("REQ-10", "   ", "en")

    # Vector dimension mismatch validation
    with pytest.raises(ValueError, match="dimension mismatch"):
        QueryEmbedder.validate_vector([0.1, 0.2], 384, "test_model")

    # Invalid numeric value validation
    with pytest.raises(ValueError, match="invalid numeric value"):
        QueryEmbedder.validate_vector([float("nan")] * 384, 384, "test_model")


@pytest.mark.asyncio
async def test_search_part5_qdrant_retriever_empty_authorized_scope():
    from backend.search.retrieval.qdrant_retriever import QdrantRetriever
    retriever = QdrantRetriever(client=None)
    res = await retriever.retrieve(
        request_id="REQ-EMPTY-SCOPE",
        query_vector=[0.1] * 384,
        vector_name="text_english_vector",
        authorized_document_ids=[],
    )
    assert res.retrieval_status == "COMPLETED"
    assert res.result_count == 0
    assert len(res.results) == 0


@pytest.mark.asyncio
async def test_search_part5_qdrant_retriever_scoped_search(in_memory_qdrant: AsyncQdrantClient):
    from qdrant_client import models
    from backend.search.retrieval.qdrant_retriever import QdrantRetriever

    collection = "test_scoped_financial_documents"
    await in_memory_qdrant.create_collection(
        collection_name=collection,
        vectors_config={
            "text_arabic_vector": models.VectorParams(size=4, distance=models.Distance.COSINE),
            "text_english_vector": models.VectorParams(size=4, distance=models.Distance.COSINE),
        },
    )

    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    # Insert chunk for Doc A (authorized)
    await in_memory_qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text_arabic_vector": [0.1, 0.2, 0.3, 0.4],
                    "text_english_vector": [0.5, 0.5, 0.5, 0.5],
                },
                payload={
                    "chunk_id": "CH-A1",
                    "document_id": doc_a,
                    "text_english": "Doc A April revenue 50000 AED",
                    "text_arabic": "إيرادات أبريل 50000 درهم",
                    "source": {"file_name": "DocA.pdf", "page": 2, "table_index": 1},
                },
            ),
            # Insert chunk for Doc B (unauthorized)
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text_arabic_vector": [0.1, 0.2, 0.3, 0.4],
                    "text_english_vector": [0.5, 0.5, 0.5, 0.5],
                },
                payload={
                    "chunk_id": "CH-B1",
                    "document_id": doc_b,
                    "text_english": "Doc B Confidential profit 99999 USD",
                    "text_arabic": "أرباح سرية 99999 دولار",
                    "source": {"file_name": "DocB.pdf", "page": 1},
                },
            ),
        ],
    )

    retriever = QdrantRetriever(client=in_memory_qdrant, collection_name=collection)

    # Query scoped exclusively to Doc A
    res = await retriever.retrieve(
        request_id="REQ-DOC-A",
        query_vector=[0.5, 0.5, 0.5, 0.5],
        vector_name="text_english_vector",
        authorized_document_ids=[doc_a],
        top_k=5,
    )
    assert res.retrieval_status == "COMPLETED"
    assert res.result_count == 1
    assert res.results[0].chunk_id == "CH-A1"
    assert res.results[0].document_id == doc_a
    assert "Doc A April revenue" in res.results[0].text_english
    assert res.results[0].source["file_name"] == "DocA.pdf"
    assert res.results[0].retrieval_score > 0.9


@pytest.mark.asyncio
async def test_search_part5_qdrant_retriever_arabic_vector_query(in_memory_qdrant: AsyncQdrantClient):
    from qdrant_client import models
    from backend.search.retrieval.qdrant_retriever import QdrantRetriever

    collection = "test_ar_col"
    await in_memory_qdrant.create_collection(
        collection_name=collection,
        vectors_config={
            "text_arabic_vector": models.VectorParams(size=4, distance=models.Distance.COSINE),
            "text_english_vector": models.VectorParams(size=4, distance=models.Distance.COSINE),
        },
    )

    doc_id = str(uuid.uuid4())
    await in_memory_qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text_arabic_vector": [0.0, 1.0, 0.0, 0.0],
                    "text_english_vector": [0.1, 0.1, 0.1, 0.1],
                },
                payload={
                    "chunk_id": "CH-AR-1",
                    "document_id": doc_id,
                    "text_arabic": "إجمالي المصاريف التشغيلية",
                    "text_english": "Total operating expenses",
                    "source": {"file_name": "Report_AR.pdf", "page": 5},
                },
            )
        ],
    )

    retriever = QdrantRetriever(client=in_memory_qdrant, collection_name=collection)
    res = await retriever.retrieve(
        request_id="REQ-AR-1",
        query_vector=[0.0, 1.0, 0.0, 0.0],
        vector_name="text_arabic_vector",
        authorized_document_ids=[doc_id],
        top_k=1,
    )
    assert res.retrieval_status == "COMPLETED"
    assert res.result_count == 1
    assert res.results[0].chunk_id == "CH-AR-1"
    assert res.results[0].text_arabic == "إجمالي المصاريف التشغيلية"
    assert res.results[0].retrieval_score >= 0.99


@pytest.mark.asyncio
async def test_search_part5_qdrant_retriever_handles_missing_collection(in_memory_qdrant: AsyncQdrantClient):
    from backend.search.retrieval.qdrant_retriever import QdrantRetriever
    retriever = QdrantRetriever(client=in_memory_qdrant, collection_name="ghost_collection_does_not_exist")
    res = await retriever.retrieve(
        request_id="REQ-ERR",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        vector_name="text_english_vector",
        authorized_document_ids=[str(uuid.uuid4())],
    )
    assert res.retrieval_status == "FAILED"
    assert res.result_count == 0


def test_search_part6_reranker_boosts_exact_financial_tokens():
    from backend.search.api.schemas import RetrievedCandidate
    from backend.search.reranking.reranker import ResultReranker

    c1 = RetrievedCandidate(
        chunk_id="CH-GENERIC",
        document_id="DOC-1",
        retrieval_score=0.70,
        text_english="General overview of corporate financial disclosures.",
        text_arabic="نظرة عامة على الإفصاحات المالية للشركة.",
        original_text="General overview of corporate financial disclosures.",
        original_language="en",
        source={"file_name": "Report.pdf", "page": 1},
    )

    c2 = RetrievedCandidate(
        chunk_id="CH-EXACT",
        document_id="DOC-1",
        retrieval_score=0.65,
        text_english="April Revenue reached 50,000 AED in Dubai.",
        text_arabic="وصلت إيرادات أبريل إلى 50,000 درهم إماراتي في دبي.",
        original_text="April Revenue reached 50,000 AED in Dubai.",
        original_language="en",
        source={"file_name": "Report.pdf", "page": 4},
    )

    query = "What was April revenue in AED?"
    financial_tokens = ["April", "AED", "50,000"]

    res = ResultReranker.rerank_and_filter(
        request_id="REQ-RERANK-1",
        query=query,
        candidates=[c1, c2],
        financial_tokens=financial_tokens,
    )

    assert res.reranking_status == "COMPLETED"
    assert res.result_count == 2
    # c2 should be boosted to rank 1 because of exact tokens ("April", "AED", "50,000")
    assert res.results[0].chunk_id == "CH-EXACT"
    assert res.results[0].relevance_rank == 1
    assert res.results[0].relevance_score > res.results[1].relevance_score
    # Original retrieval score must be strictly preserved
    assert res.results[0].retrieval_score == 0.65


def test_search_part6_reranker_threshold_filtering_and_deduplication():
    from backend.search.api.schemas import RetrievedCandidate
    from backend.search.reranking.reranker import ResultReranker

    # Duplicate chunks with same chunk_id
    c1 = RetrievedCandidate(
        chunk_id="CH-DUP",
        document_id="DOC-1",
        retrieval_score=0.85,
        text_english="Quarterly gross profit 100,000 USD",
        text_arabic="",
        original_text="Quarterly gross profit 100,000 USD",
        original_language="en",
    )
    c2 = RetrievedCandidate(
        chunk_id="CH-DUP",
        document_id="DOC-1",
        retrieval_score=0.82,
        text_english="Quarterly gross profit 100,000 USD",
        text_arabic="",
        original_text="Quarterly gross profit 100,000 USD",
        original_language="en",
    )
    # Irrelevant chunk with very low score
    c3 = RetrievedCandidate(
        chunk_id="CH-LOW",
        document_id="DOC-1",
        retrieval_score=0.10,
        text_english="Notice of annual shareholder meeting venue and parking instructions.",
        text_arabic="",
        original_text="Notice of annual shareholder meeting venue.",
        original_language="en",
    )

    res = ResultReranker.rerank_and_filter(
        request_id="REQ-RERANK-2",
        query="What is the gross profit in USD?",
        candidates=[c1, c2, c3],
        financial_tokens=["USD"],
        min_score_threshold=0.35,
    )

    # Deduped CH-DUP (only 1 retained) and CH-LOW filtered out
    assert res.result_count == 1
    assert res.results[0].chunk_id == "CH-DUP"
    assert res.filtered_out_count == 1


def test_search_part7_context_builder_language_selection():
    from backend.search.api.schemas import RetrievedCandidate
    from backend.search.context.builder import ContextBuilder

    cand = RetrievedCandidate(
        chunk_id="CH-BILINGUAL",
        document_id="DOC-99",
        retrieval_score=0.90,
        relevance_rank=1,
        relevance_score=0.92,
        text_english="Total Operating Revenue: 5,000,000 USD",
        text_arabic="إجمالي الإيرادات التشغيلية: 5,000,000 دولار أمريكي",
        original_text="Total Operating Revenue: 5,000,000 USD",
        original_language="en",
        source={"file_name": "Annual_2025.pdf", "page": 12},
    )

    # 1. Arabic response language -> selects Arabic text
    res_ar = ContextBuilder.build_context(
        request_id="REQ-CTX-AR",
        query="ما هو إجمالي الإيرادات؟",
        response_language="ar",
        candidates=[cand],
    )
    assert res_ar.chunk_count == 1
    assert res_ar.context[0].text_for_llm == "إجمالي الإيرادات التشغيلية: 5,000,000 دولار أمريكي"
    assert "إجمالي الإيرادات التشغيلية" in res_ar.formatted_context_str
    assert res_ar.context_token_count > 0

    # 2. English response language -> selects English text
    res_en = ContextBuilder.build_context(
        request_id="REQ-CTX-EN",
        query="What was total operating revenue?",
        response_language="en",
        candidates=[cand],
    )
    assert res_en.chunk_count == 1
    assert res_en.context[0].text_for_llm == "Total Operating Revenue: 5,000,000 USD"
    assert "Total Operating Revenue" in res_en.formatted_context_str


def test_search_part7_context_builder_budget_enforcement():
    from backend.search.api.schemas import RetrievedCandidate
    from backend.search.context.builder import ContextBuilder

    # Create 5 small candidate chunks (~30 tokens each)
    cands = [
        RetrievedCandidate(
            chunk_id=f"CH-{i}",
            document_id="DOC-BUDGET",
            retrieval_score=0.80 - (i * 0.05),
            relevance_rank=i,
            relevance_score=0.80 - (i * 0.05),
            text_english=f"Paragraph {i}: Quarterly revenue was reported at 50,000 USD with healthy margins.",
            text_arabic="",
            original_text=f"Paragraph {i}: Quarterly revenue was reported at 50,000 USD with healthy margins.",
            original_language="en",
            source={"file_name": "Report.pdf", "page": i},
        )
        for i in range(1, 6)
    ]

    # Constrain to budget of 120 tokens (allows ~2 chunks plus headers)
    res = ContextBuilder.build_context(
        request_id="REQ-CTX-BUDGET",
        query="Tell me about financial notes",
        response_language="en",
        candidates=cands,
        max_context_tokens=120,
    )

    assert 1 <= res.chunk_count < len(cands)
    assert res.context_token_count <= 150


@pytest.mark.asyncio
async def test_search_part9_answer_generator_empty_context():
    from backend.search.generation.answer_generator import AnswerGenerator

    gen = AnswerGenerator()
    res = await gen.generate_draft_answer(
        request_id="REQ-GEN-EMPTY",
        original_query="What is the net profit?",
        response_language="en",
        formatted_context_str="",
        context_chunks=[],
    )
    assert res.generation_status == "INSUFFICIENT_EVIDENCE"
    assert "Insufficient evidence" in res.answer
    assert len(res.citations) == 0


@pytest.mark.asyncio
async def test_search_part9_answer_generator_grounded_mock():
    from backend.search.context.builder import ContextChunk
    from backend.search.generation.answer_generator import AnswerGenerator

    chunk = ContextChunk(
        chunk_id="CH-TEST-001",
        document_id="DOC-TEST-1",
        original_language="en",
        text_for_llm="April revenue is 50,000 AED",
        source={"file_name": "April_Report.pdf", "page": 4},
        relevance_rank=1,
        relevance_score=0.95,
    )

    gen = AnswerGenerator()
    res = await gen.generate_draft_answer(
        request_id="REQ-GEN-GROUNDED",
        original_query="What was April revenue in AED?",
        response_language="en",
        formatted_context_str="[Evidence Chunk CH-001]\nContent:\nApril revenue is 50,000 AED",
        context_chunks=[chunk],
    )
    assert res.generation_status == "COMPLETED"
    assert "50,000 AED" in res.answer
    assert "[Chunk CH-001]" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].chunk_id == "CH-TEST-001"
    assert res.citations[0].file_name == "April_Report.pdf"


def test_search_part10_citation_verifier_passes_on_valid_figures():
    from backend.search.api.schemas import Citation
    from backend.search.context.builder import ContextChunk
    from backend.search.validation.citation_verifier import CitationAndEvidenceVerifier

    chunk = ContextChunk(
        chunk_id="CH-VAL-01",
        document_id="DOC-VAL-1",
        original_language="en",
        text_for_llm="Q1 operating revenue totaled 50000 USD",
        source={"file_name": "Q1.pdf", "page": 2},
        relevance_rank=1,
        relevance_score=0.92,
    )
    citation = Citation(
        chunk_id="CH-VAL-01",
        document_id="DOC-VAL-1",
        file_name="Q1.pdf",
        page=2,
    )
    answer = "According to records, Q1 operating revenue was 50000 USD [Chunk CH-001]."

    ver = CitationAndEvidenceVerifier.validate_answer_and_citations(
        request_id="REQ-VER-PASS",
        answer=answer,
        context_chunks=[chunk],
        citations=[citation],
    )
    assert ver.validation_status == "PASSED"
    assert ver.citation_validation == "PASSED"
    assert ver.evidence_validation == "PASSED"
    assert ver.ready_to_return is True
    assert len(ver.unsupported_claims) == 0


def test_search_part10_citation_verifier_fails_on_hallucinated_figures():
    from backend.search.api.schemas import Citation
    from backend.search.context.builder import ContextChunk
    from backend.search.validation.citation_verifier import CitationAndEvidenceVerifier

    chunk = ContextChunk(
        chunk_id="CH-VAL-02",
        document_id="DOC-VAL-2",
        original_language="en",
        text_for_llm="April revenue is 50000 AED",
        source={"file_name": "Report.pdf", "page": 1},
        relevance_rank=1,
        relevance_score=0.88,
    )
    citation = Citation(
        chunk_id="CH-VAL-02",
        document_id="DOC-VAL-2",
        file_name="Report.pdf",
        page=1,
    )
    # Answer hallucinates 999999 USD which is completely absent from context
    hallucinated_answer = "Total revenue reached 999999 USD in April [Chunk CH-001]."

    ver = CitationAndEvidenceVerifier.validate_answer_and_citations(
        request_id="REQ-VER-FAIL",
        answer=hallucinated_answer,
        context_chunks=[chunk],
        citations=[citation],
    )
    assert ver.validation_status == "FAILED"
    assert ver.evidence_validation == "FAILED"
    assert ver.ready_to_return is False
    assert any("999999" in claim for claim in ver.unsupported_claims)
    assert "Audit Warning" in ver.validated_answer


def test_search_part10_citation_verifier_fails_on_ghost_citation():
    from backend.search.api.schemas import Citation
    from backend.search.context.builder import ContextChunk
    from backend.search.validation.citation_verifier import CitationAndEvidenceVerifier

    chunk = ContextChunk(
        chunk_id="CH-REAL-01",
        document_id="DOC-1",
        original_language="en",
        text_for_llm="Net income is 20000 AED",
        source={"file_name": "Doc.pdf", "page": 1},
        relevance_rank=1,
        relevance_score=0.90,
    )
    ghost_citation = Citation(
        chunk_id="CH-GHOST-999",  # Does not exist in context!
        document_id="DOC-1",
        file_name="Ghost.pdf",
    )
    answer = "Net income is 20000 AED [Chunk CH-GHOST-999]."

    ver = CitationAndEvidenceVerifier.validate_answer_and_citations(
        request_id="REQ-VER-GHOST",
        answer=answer,
        context_chunks=[chunk],
        citations=[ghost_citation],
    )
    assert ver.validation_status == "FAILED"
    assert ver.citation_validation == "FAILED"
    assert ver.ready_to_return is False
    assert any("CH-GHOST-999" in claim for claim in ver.unsupported_claims)


def test_search_part8_calculation_engine_sum():
    from decimal import Decimal
    from backend.search.calculation.engine import FinancialCalculationEngine
    from backend.search.context.builder import ContextChunk

    c1 = ContextChunk(
        chunk_id="CH-1",
        document_id="DOC-1",
        original_language="en",
        text_for_llm="Q1 Revenue: 500000.00 AED",
        source={"file_name": "Report.pdf"},
        relevance_rank=1,
        relevance_score=0.95,
    )
    c2 = ContextChunk(
        chunk_id="CH-2",
        document_id="DOC-1",
        original_language="en",
        text_for_llm="Q2 Revenue: 620000.50 AED",
        source={"file_name": "Report.pdf"},
        relevance_rank=2,
        relevance_score=0.90,
    )

    query = "What is the total revenue for Q1 and Q2 in AED?"
    calc = FinancialCalculationEngine.compute(
        request_id="REQ-CALC-1",
        query=query,
        context_chunks=[c1, c2],
    )
    assert calc.performed is True
    assert calc.operation == "SUM"
    assert calc.result_value == Decimal("1120000.50")
    assert calc.currency == "AED"
    assert "1120000.50 AED" in calc.formula_str
    assert calc.currency_mismatch is False


def test_search_part8_calculation_engine_currency_mismatch_guard():
    from backend.search.calculation.engine import FinancialCalculationEngine
    from backend.search.context.builder import ContextChunk

    c1 = ContextChunk(
        chunk_id="CH-1",
        document_id="DOC-1",
        original_language="en",
        text_for_llm="Revenue in Dubai: 50000 AED",
        source={"file_name": "Report.pdf"},
        relevance_rank=1,
        relevance_score=0.95,
    )
    c2 = ContextChunk(
        chunk_id="CH-2",
        document_id="DOC-1",
        original_language="en",
        text_for_llm="Revenue in New York: 10000 USD",
        source={"file_name": "Report.pdf"},
        relevance_rank=2,
        relevance_score=0.90,
    )

    query = "What is the combined total revenue?"
    calc = FinancialCalculationEngine.compute(
        request_id="REQ-CALC-2",
        query=query,
        context_chunks=[c1, c2],
    )
    assert calc.performed is False
    assert calc.currency_mismatch is True
    assert "mismatched currencies" in calc.explanation


def test_search_part8_calculation_engine_arabic_intent():
    from backend.search.calculation.engine import FinancialCalculationEngine
    assert FinancialCalculationEngine.has_calculation_intent("ما هو إجمالي الإيرادات لشهر أبريل؟") is True
    assert FinancialCalculationEngine.has_calculation_intent("ما هو مجموع الأرباح؟") is True
    assert FinancialCalculationEngine.has_calculation_intent("ما هو الفرق بين الربعين؟") is True
    assert FinancialCalculationEngine.has_calculation_intent("أين يقع مقر الشركة الرئيسي؟") is False


@pytest.mark.asyncio
async def test_search_part11_pipeline_orchestrator_e2e(db_session: AsyncSession):
    from backend.search.orchestration.pipeline import SearchPipelineOrchestrator
    from backend.search.api.schemas import QueryRequest

    # Test orchestrator with query for empty scope
    orchestrator = SearchPipelineOrchestrator(db=db_session)
    request = QueryRequest(
        query="What is the total quarterly revenue?",
        document_names=["NonExistentDocument.pdf"],
    )

    # Unknown document raises 404 from Step 2 access resolver
    with pytest.raises(Exception):
        await orchestrator.execute_query(request_id="REQ-ORCH-1", request=request)






