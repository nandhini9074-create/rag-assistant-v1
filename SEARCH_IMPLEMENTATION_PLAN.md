# Part-by-Part Implementation Plan: Financial RAG Search & Retrieval Flow

## Objective
Implement Method 1: Retrieval & Question Answering Workflow for the Arabic–English Financial Document Chatbot.
All implementation code will reside strictly inside `backend/search/`, with zero modifications to any files under `backend/ingestion/`.

---

## Architecture Overview & Ingestion Synergy
In the ingestion pipeline, each chunk in Qdrant (`collection: financial_documents`) contains:
- **`text_english_vector`**: 384-dimensional vector encoded via `sentence-transformers/all-MiniLM-L6-v2`.
- **`text_arabic_vector`**: 768-dimensional vector encoded via `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2`.
- **Payload**: `chunk_id`, `document_id`, `original_language`, `original_text`, `text_english`, `text_arabic`, `source` (file_name, page, table_index, etc.).

When a query arrives in Arabic or English:
1. Detect query language (`ar` vs `en`).
2. Generate matching embedding:
   - Arabic query &rarr; Arabic model (`dim=768`) &rarr; search against `text_arabic_vector`.
   - English query &rarr; English model (`dim=384`) &rarr; search against `text_english_vector`.
   - Supports cross-lingual query mapping: English queries can retrieve against both spaces or targeted dual-search if needed.
3. Apply Qdrant filter on `document_id` matching user's authorized scope.
4. Rerank candidates by semantic score and financial entity density.
5. Build context within token budget.
6. Generate answer using GPT-4o (with automatic Gemini fallback).
7. Validate citations, claim veracity, and financial figures (numbers, dates, currency).
8. Return typed response conforming to `POST /api/v1/chat/query`.

---

## Directory Structure
```
backend/search/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic v2 schemas: QueryRequest, QueryResponse, Citation, AnswerMetadata
│   └── router.py               # Endpoint POST /api/v1/chat/query
├── access_control/
│   ├── __init__.py
│   └── resolver.py             # Document name resolution & user authorization against PostgreSQL
├── language/
│   ├── __init__.py
│   └── detector.py             # Query language detection (en vs ar, dialect safeguards)
├── preprocessing/
│   ├── __init__.py
│   └── query_cleaner.py        # Financial entity, numbers, and currency preservation
├── embeddings/
│   ├── __init__.py
│   └── query_embedder.py       # Query vector generator matching dual Qdrant vector spaces
├── retrieval/
│   ├── __init__.py
│   └── qdrant_retriever.py     # Qdrant search with authorized_document_ids payload filters
├── reranking/
│   ├── __init__.py
│   └── reranker.py             # Relevance scoring, deduplication, and score thresholding
├── context/
│   ├── __init__.py
│   └── builder.py              # Token-bounded context assembly & citation metadata mapping
├── generation/
│   ├── __init__.py
│   └── answer_generator.py     # Evidence-grounded generation (GPT-4o primary, Gemini fallback)
├── validation/
│   ├── __init__.py
│   └── citation_verifier.py    # 2-stage verification: citation existence & financial number fidelity
├── calculation/
│   ├── __init__.py
│   └── engine.py               # Deterministic Decimal calculation for totals, averages, and margins
└── orchestration/
    ├── __init__.py
    └── pipeline.py             # Steps 1 to 11 coordinator
```

---

## Step-by-Step Implementation Breakdown

### Part 1: Typed Request & Response Schemas (`backend/search/api/schemas.py`)
- `QueryRequest`:
  - `query: str` (min_length=1, max_length=2000)
  - `document_names: Optional[list[str]] = None`
  - `conversation_id: Optional[str] = None`
  - `top_k: int = 10`
- `Citation`:
  - `chunk_id: str`
  - `document_id: str`
  - `file_name: str`
  - `page: Optional[int]`
  - `table_index: Optional[int]`
- `AnswerMetadata`:
  - `citation_validation: str` ("PASSED" | "FAILED" | "WARNING")
  - `evidence_validation: str` ("PASSED" | "FAILED" | "INSUFFICIENT_EVIDENCE")
  - `calculation_performed: bool`
  - `unsupported_claims: list[str]`
- `QueryResponse`:
  - `request_id: str` (e.g. `REQ-<uuid>`)
  - `conversation_id: Optional[str]`
  - `status: str` ("COMPLETED" | "INSUFFICIENT_EVIDENCE" | "FAILED")
  - `answer: str`
  - `response_language: str` ("ar" | "en")
  - `citations: list[Citation]`
  - `answer_metadata: AnswerMetadata`

---

### Part 2: Document Resolution & Access Control (`backend/search/access_control/resolver.py`)
- Query PostgreSQL `documents` table by `file_name`.
- Check `status == ACTIVE`. If document is `PENDING` or `FAILED`, exclude with an informative warning.
- Check user authorization (or return all active documents if `document_names` is null).
- Handle ambiguous filenames and unknown filenames by returning clear 404/400 errors.

---

### Part 3: Query Language Detection (`backend/search/language/detector.py`) [COMPLETED & VERIFIED]
- Detect whether query is Arabic (`ar`) or English (`en`) using Lingua with Arabic script regex fallback (`[\u0600-\u06FF]`).
- Confidence scoring, mixed Arabic-English code-switching handling, symbol-only fallback.

---

### Part 4: Query Preparation & Embedding Generation (`backend/search/preprocessing/query_cleaner.py`, `backend/search/embeddings/query_embedder.py`) [COMPLETED & VERIFIED]
- **Step 4: Query Preprocessor (`QueryPreprocessor`)**:
  - Normalizes excessive whitespace and zero-width artifacts while strictly preserving Arabic and English financial syntax.
  - Preserves currency symbols/codes (AED, USD, SAR, etc.), percentages (%), signs (+/-), decimal numbers, quarters (Q1-Q4), and fiscal years.
  - Extracts financial tokens for downstream relevance reranking and arithmetic verification.
- **Step 5: Query Embedder (`QueryEmbedder`)**:
  - Language-directed vector mapping to compatible Qdrant vector space:
    - Arabic (`ar`) &rarr; `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` &rarr; `text_arabic_vector` (`dim=768`).
    - English (`en`) &rarr; `sentence-transformers/all-MiniLM-L6-v2` &rarr; `text_english_vector` (`dim=384`).
  - Dual-vector generation support for cross-lingual searches.
  - Vector numerical validation (zero NaN/Inf, strict dimension verification).
  - In-memory cached model loading and deterministic testing mode.

---

### Part 5: Qdrant Vector Retrieval (`backend/search/retrieval/qdrant_retriever.py`) [COMPLETED & VERIFIED]
- **Step 6: Qdrant Retriever (`QdrantRetriever`)**:
  - Connects asynchronously to Qdrant cluster via `query_points`.
  - Security Enforced: Empty authorized document scope returns 0 candidates immediately without executing an open query.
  - Constructs `document_id` matching filter (`MatchAny`) across user's authorized documents.
  - Performs vector search against the language-directed vector (`text_arabic_vector` or `text_english_vector`).
  - Maps payload metadata into structured `RetrievedCandidate` instances (scores, English/Arabic text, chunk_id, document_id, and source metadata).
  - Handles missing collections and network errors gracefully returning `retrieval_status="FAILED"`.

---

### Part 6: Reranking & Relevance Filtering (`backend/search/reranking/reranker.py`) [COMPLETED & VERIFIED]
- **Step 7: Result Reranker (`ResultReranker`)**:
  - Scores candidates by combining dense vector similarity with lexical and exact financial token matching (currencies, numbers, quarters, fiscal years).
  - Preserves original `retrieval_score` separate from `relevance_score`.
  - Deduplicates identical chunks by `chunk_id` and text signature.
  - Filters low-scoring candidates using configurable `min_score_threshold` (default 0.35).
  - Assigns calibrated `relevance_rank` (1, 2, 3...) and caps output to `max_candidates`.

---

### Part 7: Context Assembly (`backend/search/context/builder.py`) [COMPLETED & VERIFIED]
- **Step 8: Context Builder (`ContextBuilder`)**:
  - Assembles evidence in strict `relevance_rank` order into structured markdown blocks.
  - Dynamically selects optimal text representation (`text_arabic` when user queries in Arabic; `text_english` when English; preserving original text fallback).
  - Token counting with calibrated Arabic/English heuristics matching GPT-4o / multilingual tokenizers.
  - Strict token budgeting enforcing `max_context_tokens` (default 4000).
  - Injects security boundary marking evidence as untrusted document data for reference only.

---

### Part 8: Financial Calculation Engine (`backend/search/calculation/engine.py`)
- Detect calculation intent (sum, total, difference, margin).
- Extract monetary values using Python `Decimal` for precision.
- Guard against currency mismatch (e.g. cannot sum USD and AED without exchange rate).

---

### Part 9: Answer Generation (`backend/search/generation/answer_generator.py`) [COMPLETED & VERIFIED]
- **Step 9: Answer Generator (`AnswerGenerator`)**:
  - Grounded financial prompt strictly preventing false aggregation and number hallucinations.
  - Multi-tier LLM fallback (GPT-4o primary &rarr; Gemini 3.5 Flash Lite fallback &rarr; extractive fallback).
  - Explicit chunk references in output (`[Chunk CH-001]`).
  - Response language fidelity (`ar` or `en`).
  - Safe insufficient-evidence acknowledgment.

---

### Part 10: Evidence & Citation Verification (`backend/search/validation/citation_verifier.py`) [COMPLETED & VERIFIED]
- **Step 10: Citation & Evidence Verifier (`CitationAndEvidenceVerifier`)**:
  - Two-stage verification:
    1. **Citation Integrity**: Verifies that every cited chunk ID and document ID exists in the retrieved context.
    2. **Numerical Fidelity**: Deterministically parses all financial numbers/amounts in the answer and verifies presence in the cited evidence chunks.
  - Detects fabricated numbers, ghost citations, and unverified claims.
  - Appends audit warnings and flags `unsupported_claims` if validation fails.

---

### Part 11: Orchestration & FastAPI Router (`backend/search/orchestration/pipeline.py`, `backend/search/api/router.py`)
- Endpoint: `POST /api/v1/chat/query`
- Execute Steps 1 to 11 synchronously and return `QueryResponse`.
- Register router in `backend/main.py`.

---

### Part 12: Comprehensive Automated Test Suite (`tests/test_search_flow.py`)
- Unit tests for:
  - English and Arabic query routing.
  - Document access resolution and authorization filtering.
  - Dual vector retrieval against Qdrant.
  - Number and citation validation.
  - End-to-end question answering with citations.
