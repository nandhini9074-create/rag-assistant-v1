# Multilingual Financial RAG Ingestion Pipeline — Comprehensive Implementation Plan

## 1. Executive Summary & Objective

This document defines the complete architectural and implementation plan for the document ingestion pipeline of a multilingual financial Retrieval-Augmented Generation (RAG) system.

### Key Objectives:
- Implement all **12 sequential ingestion steps** with strict input/output typing and fault isolation.
- Use **PostgreSQL (SQLAlchemy 2.x, Alembic)** as the source of truth for documents, jobs, and chunk manifests.
- Use **private deterministic file storage** without exposing file paths or storing `file_uri` columns in the database.
- Process documents asynchronously using a background queue (**Celery + Redis**).
- Parse diverse financial documents: **PDF** (PyMuPDF text or Azure AI Document Intelligence OCR for scans), **DOCX** (python-docx), **XLSX** (openpyxl), and **CSV** (Python `csv`).
- Preserve **financial tables, signs, precision, currencies, and dates** without destructive flattening.
- Detect language using **Lingua** (`en` and `ar`).
- Produce bilingual text using **OpenAI GPT-4o** with strict financial entity preservation.
- Generate embeddings using **separate Sentence Transformers models** for English and Arabic (strictly no OpenAI embeddings):
  - **English**: Configurable Sentence Transformers model (e.g., `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`) with dynamic dimension validation.
  - **Arabic**: `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` with pinned revision and dynamic dimension validation.
- Store both vectors in a **single Qdrant point** with named vectors: `text_english_vector` and `text_arabic_vector`.
- Perform **deep Qdrant verification** before marking chunks `VERIFIED` and documents `ACTIVE`.
- Implement **scheduled/CLI reconciliation** to detect and repair drift, orphaned vectors, and stuck jobs.
- Note: Chat APIs, question answering, retrieval, and generation are explicitly out of scope for this task.

---

## 2. High-Level Architecture & End-to-End Data Flow

```mermaid
flowchart TD
    Client([Client / API Consumer]) -->|POST /api/v1/documents| S1[Step 1: File Validation & Registration]
    S1 -->|Store File Privately| PrivStore[(Private Storage)]
    S1 -->|Insert UPLOADED & QUEUED| PG[(PostgreSQL)]
    S1 -->|Enqueue Job| Queue[(Celery / Redis)]
    
    Queue --> Worker[Ingestion Worker]
    Worker --> S2[Step 2: File Routing]
    
    subgraph S3 [Step 3: Content Extraction]
        S2 -->|Text PDF| PyMuPDF[PyMuPDF]
        S2 -->|Scanned PDF| AzureOCR[Azure Document Intelligence]
        S2 -->|DOCX| PyDocx[python-docx]
        S2 -->|XLSX| OpenPyXL[openpyxl]
        S2 -->|CSV| PyCSV[csv module]
    end
    
    S3 --> S4[Step 4: Data Normalization]
    S4 --> S5[Step 5: Adaptive Chunking & Manifest Generation]
    S5 -->|Persist Chunk Manifest PENDING| PG
    
    S5 --> S6[Step 6: Language Detection - Lingua]
    S6 --> S7[Step 7: Bilingual Translation - GPT-4o]
    
    subgraph S8 [Step 8: Dual Sentence Transformers]
        S7 --> ST_EN[English Service: Configurable ST Model]
        S7 --> ST_AR[Arabic Service: Arabic-Triplet-Matryoshka-V2]
    end
    
    ST_EN -->|text_english_vector| S9[Step 9: Dual Named Vector Storage]
    ST_AR -->|text_arabic_vector| S9
    S9 -->|Upsert Point| Qdrant[(Qdrant: financial_documents)]
    
    S9 --> S10[Step 10: Deep Qdrant Verification]
    Qdrant -->|Retrieve & Verify Point| S10
    S10 -->|Update Chunk VERIFIED| PG
    
    S10 --> S11[Step 11: Document Activation & Completion]
    S11 -->|Set ACTIVE / COMPLETED| PG
    
    Cron[Reconciliation Engine / CLI] --> S12[Step 12: Ingestion Health & Reconciliation]
    S12 <--> PG
    S12 <--> Qdrant
```

---

## 3. Technology Stack & Component Specifications

| Purpose | Tool / Library | Version / Details |
|---|---|---|
| **API & Validation** | FastAPI, Pydantic v2 | Python 3.12, strict schemas, multipart upload handling |
| **Database** | PostgreSQL 16 | Source of truth, foreign keys, unique constraints, JSONB |
| **ORM & Migrations** | SQLAlchemy 2.x, Alembic | Async engine via `psycopg` (v3) |
| **Background Processing** | Celery + Redis | Asynchronous job execution with retry policies and task tracking |
| **PDF Extraction** | PyMuPDF (`fitz`) | Text extraction, page layout inspection, table discovery |
| **Scanned PDF OCR** | Azure AI Document Intelligence | `azure-ai-documentintelligence` with `prebuilt-layout` model |
| **DOCX Extraction** | `python-docx` | Headings, paragraphs, table extraction with cell coordinates |
| **XLSX Extraction** | `openpyxl` | Worksheets, formula value resolution, cell coordinates |
| **CSV Extraction** | Python standard `csv` | `csv.Sniffer` for delimiter detection, row-level tracking |
| **Language Detection** | `lingua-language-detector` | High-accuracy detection for English and Arabic with confidence |
| **Translation** | OpenAI GPT-4o | Financial entity preservation prompts and validation |
| **English Embeddings** | Sentence Transformers | Configurable via `ENGLISH_EMBEDDING_MODEL` (e.g. `sentence-transformers/all-MiniLM-L6-v2`) |
| **Arabic Embeddings** | Sentence Transformers | `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` with pinned revision |
| **Vector Database** | Qdrant | Named vectors: `text_english_vector` and `text_arabic_vector` in collection `financial_documents` |
| **Testing** | pytest, pytest-asyncio, HTTPX | Unit tests, mock tests, full pipeline integration tests |
| **Local Services** | Docker Compose | PostgreSQL 16, Redis 7, Qdrant |

---

## 4. Database Schema Requirements (PostgreSQL)

### 4.1 Table: `documents`
Stores document metadata and lifecycle status.
- `document_id`: `UUID(as_uuid=True)`, Primary Key, default `uuid4`.
- `file_name`: `String(255)`, Not Null.
- `file_type`: `String(32)`, Not Null (normalized: `'pdf'`, `'docx'`, `'xlsx'`, `'csv'`).
- `status`: `Enum('UPLOADED', 'PROCESSING', 'ACTIVE', 'FAILED', name='document_status')`, default `'UPLOADED'`.
- `total_chunks`: `Integer`, default `0`, Check Constraint `>= 0`.
- `uploaded_by`: `String(128)`, Nullable.
- `file_size_bytes`: `BigInteger`, Nullable, Check Constraint `>= 0`.
- `failure_reason`: `Text`, Nullable.
- `created_at`: `DateTime(timezone=True)`, default `func.now()`.
- `updated_at`: `DateTime(timezone=True)`, default `func.now()`, onupdate `func.now()`.
- **Strict Rule:** No `file_uri` or local storage path column is included.

### 4.2 Table: `ingestion_jobs`
Tracks each ingestion attempt and step execution.
- `job_id`: `UUID(as_uuid=True)`, Primary Key, default `uuid4`.
- `document_id`: `UUID(as_uuid=True)`, ForeignKey(`documents.document_id`, ondelete='CASCADE'), Not Null.
- `status`: `Enum('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', name='job_status')`, default `'QUEUED'`.
- `current_step`: `String(64)`, Nullable.
- `parser`: `String(64)`, Nullable (`'pymupdf'`, `'azure_ocr'`, `'python-docx'`, `'openpyxl'`, `'csv'`).
- `total_chunks`: `Integer`, default `0`.
- `processed_chunks`: `Integer`, default `0`.
- `verified_chunks`: `Integer`, default `0`.
- `attempt_number`: `Integer`, default `1`, Not Null.
- `started_at`: `DateTime(timezone=True)`, Nullable.
- `completed_at`: `DateTime(timezone=True)`, Nullable.
- `error_step`: `String(64)`, Nullable.
- `failure_reason`: `Text`, Nullable.
- `created_at`, `updated_at`: `DateTime(timezone=True)`.
- **Constraints & Indexes:**
  - Unique Constraint on `(document_id, attempt_number)`.
  - Index on `(document_id, status)`.
  - Index on `status`.

### 4.3 Table: `document_chunks`
Acts as the expected-chunk manifest and verification source of truth.
- `chunk_id`: `UUID(as_uuid=True)`, Primary Key, default `uuid4`.
- `document_id`: `UUID(as_uuid=True)`, ForeignKey(`documents.document_id`, ondelete='CASCADE'), Not Null.
- `job_id`: `UUID(as_uuid=True)`, ForeignKey(`ingestion_jobs.job_id`, ondelete='CASCADE'), Not Null.
- `chunk_index`: `Integer`, Not Null.
- `content_type`: `String(64)`, Not Null (`'financial_table'`, `'narrative_text'`, `'heading'`, `'transaction_record'`).
- `original_language`: `String(10)`, Nullable (`'en'`, `'ar'`, or uncertain).
- `translation_status`: `String(32)`, default `'PENDING'` (`'PENDING'`, `'COMPLETED'`, `'SKIPPED'`, `'FAILED'`).
- `qdrant_point_id`: `UUID(as_uuid=True)`, Not Null.
- `source_metadata`: `JSONB`, Not Null (stores page/row/cell references, model IDs, revisions, dimensions).
- `status`: `Enum('PENDING', 'PROCESSED', 'VERIFIED', 'FAILED', name='chunk_status')`, default `'PENDING'`.
- `failure_reason`: `Text`, Nullable.
- `created_at`, `updated_at`: `DateTime(timezone=True)`.
- **Constraints & Indexes:**
  - Unique Constraint on `(document_id, job_id, chunk_index)`.
  - Index on `(document_id, status)`.
  - Index on `(job_id, status)`.
  - Index on `qdrant_point_id`.

---

## 5. Detailed Step-by-Step Ingestion Pipeline (12 Steps)

### Step 1: File Validation & Registration
- **Endpoint**: `POST /api/v1/documents` (`multipart/form-data`)
- **Input**: `file: UploadFile`
- **Validation Rules**:
  - File extension normalized (`.pdf`, `.docx`, `.xlsx`, `.csv`).
  - Magic byte / MIME type validation (prevent renamed executables).
  - Size check against `MAX_UPLOAD_SIZE_MB` (default 50MB).
  - Integrity check: reject empty, encrypted (e.g. password-protected PDFs), or corrupt files.
- **Storage**:
  - Deterministic private path: `{PRIVATE_STORAGE_ROOT}/{document_id}/source.{ext}`.
  - Safe file permissions applied.
- **Persistence & Queueing**:
  - Insert `documents` row (`status = 'UPLOADED'`).
  - Insert `ingestion_jobs` row (`status = 'QUEUED'`, `attempt_number = 1`).
  - Enqueue job in Celery / Redis.
  - Rollback handler: If database insertion or queueing fails, delete the private file and return an error.
- **Output**: `HTTP 202 Accepted`
  ```json
  {
    "document_id": "018f3a2c-1234-7000-8000-000000000001",
    "job_id": "018f3a2c-1234-7000-8000-000000000002",
    "file_name": "annual_report.pdf",
    "file_type": "pdf",
    "status": "UPLOADED",
    "job_status": "QUEUED",
    "created_at": "2026-10-27T10:00:00Z"
  }
  ```

### Step 2: File Routing
- **Worker Logic**:
  - Update `ingestion_jobs`: `current_step = 'FILE_ROUTING'`, `status = 'PROCESSING'`, `started_at = now()`.
  - Inspect document content to select the appropriate parser:
    - **PDF**: Inspect page text using PyMuPDF. If usable text layer exists across pages $\rightarrow$ `pymupdf`. If scanned images with no text $\rightarrow$ `azure_ocr`. For mixed PDFs, route text pages to PyMuPDF and scanned pages to OCR.
    - **DOCX** $\rightarrow$ `python-docx`.
    - **XLSX** $\rightarrow$ `openpyxl`.
    - **CSV** $\rightarrow$ `csv`.
  - Persist `parser` and `current_step` in `ingestion_jobs`.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "job_id": "018f3a2c-...",
    "parser": "pymupdf",
    "status": "PROCESSING",
    "current_step": "FILE_ROUTING"
  }
  ```

### Step 3: Content Extraction
- **Parsers Implementation**:
  - **PyMuPDF**: Extract text blocks, detect headings based on font size/weight, extract tables via `page.find_tables()`, preserve page numbers and bounding boxes.
  - **Azure AI Document Intelligence**: Call layout model, extract text paragraphs, headings, and structured tables with row/col spans.
  - **python-docx**: Extract headings, paragraphs, and XML table structures with cell text.
  - **openpyxl**: Iterate sheets, extract tabular regions, evaluate formulas, preserve cell coordinates.
  - **csv**: Sniff delimiter, extract headers and row values.
- **Financial Table Preservation**:
  - Preserve exact values, currencies (e.g. AED, USD), signs (e.g. `(50,000)` or `-50000`), decimals, and column/row header associations.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "extracted_items": [
      {
        "type": "text",
        "text": "April revenue was AED 50000.",
        "source": { "page": 3 }
      },
      {
        "type": "table",
        "headers": ["Month", "Revenue", "Currency"],
        "rows": [["April", "50000", "AED"], ["May", "60000", "AED"]],
        "source": { "page": 4 }
      }
    ]
  }
  ```

### Step 4: Data Normalization
- **Logic**:
  - Transform parser outputs into typed Pydantic models: `NormalizedElement`.
  - Generate deterministic `element_id` (UUIDv5 of `document_id` + source coordinates + sequence index).
  - Refine content types: `table` $\rightarrow$ `financial_table`, `heading`, `narrative_text`, `transaction_record`.
  - Format financial tables into structured markdown table format (`Col1 | Col2 \n Val1 | Val2`) while preserving raw cell data in metadata.
  - Strict preservation of currency codes, transaction IDs, account numbers, and decimal precision.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "elements": [
      {
        "element_id": "018f3a2c-...",
        "content_type": "financial_table",
        "text": "Month | Revenue | Currency\nApril | 50000 | AED\nMay | 60000 | AED",
        "source": { "page": 4 }
      }
    ]
  }
  ```

### Step 5: Adaptive Chunking & Manifest Generation
- **Token Constraints**:
  - Compute safe max token limit: `safe_limit = min(en_model_max_tokens, ar_model_max_tokens) - buffer`.
  - Chunk size: configurable via `CHUNK_SIZE_TOKENS` (default 384 tokens), overlap `CHUNK_OVERLAP_TOKENS` (default 48 tokens).
- **Chunking Rules**:
  - **Headings**: Kept bound to the following text/table content.
  - **Tables**: Large tables are split into row batches with **headers repeated** on every chunk.
  - **Transactions / Key-Value Pairs**: Kept intact in the same chunk.
- **Manifest Persistence & Retry Idempotency**:
  - Generate deterministic `chunk_id` and `qdrant_point_id` (UUIDv5 from `document_id` + `chunk_index`).
  - Persist all chunks in `document_chunks` table with `status = 'PENDING'`.
  - Update `ingestion_jobs.total_chunks = len(chunks)`.
  - On retries, reuse existing chunk manifest records.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "chunks": [
      {
        "chunk_id": "018f3a2c-...",
        "chunk_index": 0,
        "element_ids": ["018f3a2c-..."],
        "content_type": "financial_table",
        "text": "Month | Revenue | Currency\nApril | 50000 | AED",
        "source": { "page": 4 }
      }
    ]
  }
  ```

### Step 6: Language Detection
- **Tool**: `lingua-language-detector` configured for English and Arabic.
- **Logic**:
  - Detect language per chunk.
  - For numeric/table-dense chunks, analyze textual labels or use document context.
  - If Lingua supplies confidence, record `confidence_score` in `source_metadata`.
  - Strict policy: If language is ambiguous or unsupported, do not guess; fail chunk with diagnostic reason.
- **Output**:
  ```json
  {
    "chunk_id": "018f3a2c-...",
    "document_id": "018f3a2c-...",
    "original_language": "en",
    "original_text": "April revenue was AED 50000.",
    "source": { "page": 3 }
  }
  ```

### Step 7: Translation & Bilingual Text Preparation
- **Tool**: OpenAI GPT-4o with strict system prompt and schema validation.
- **Logic**:
  - If `original_language == 'en'`: `text_english = original_text`, translate into `text_arabic`.
  - If `original_language == 'ar'`: `text_arabic = original_text`, translate into `text_english`.
  - Financial entity preservation: Numeric figures, decimals, negative signs, currency symbols, dates, account numbers, and table separators must remain identical.
  - Bounded exponential retries (up to 3 attempts).
  - Validation: Ensure both `text_english` and `text_arabic` are non-empty and well-formed.
  - Update `document_chunks.translation_status = 'COMPLETED'`.
- **Output**:
  ```json
  {
    "chunk_id": "018f3a2c-...",
    "document_id": "018f3a2c-...",
    "original_language": "en",
    "original_text": "April revenue was AED 50000.",
    "text_english": "April revenue was AED 50000.",
    "text_arabic": "بلغت إيرادات أبريل 50000 درهم إماراتي.",
    "translation_status": "COMPLETED",
    "source": { "page": 3 }
  }
  ```

### Step 8: Separate Sentence Transformers Embedding Generation
- **Architecture**:
  - Two distinct service classes: `EnglishEmbeddingService` and `ArabicEmbeddingService`.
  - **English Model**: Configurable via `ENGLISH_EMBEDDING_MODEL` (e.g. `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`), verified Apache 2.0 / MIT license, 512 token context.
  - **Arabic Model**: `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` with pinned Hugging Face revision.
- **Startup Dimension Validation**:
  - Models loaded and probed with sample inference at application startup.
  - Actual vector dimensions resolved dynamically (`dim_en = len(vec_en)`, `dim_ar = len(vec_ar)`).
  - Never hardcode dimensions (such as 1024 or 768) without runtime validation.
- **Inference & Validation**:
  - Generate `text_english_vector` from `text_english`.
  - Generate `text_arabic_vector` from `text_arabic`.
  - Verify vectors are non-empty numeric arrays, dimensions match validated dimensions, and no `NaN` or `Inf` exists.
  - Never pad or truncate vectors artificially.
- **Output**:
  ```json
  {
    "chunk_id": "018f3a2c-...",
    "document_id": "018f3a2c-...",
    "qdrant_point_id": "018f3a2c-...",
    "embedding_models": {
      "text_english_vector": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "e4ce9877...",
        "dimensions": 384
      },
      "text_arabic_vector": {
        "model": "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
        "revision": "f1a2b3...",
        "dimensions": 768
      }
    },
    "embeddings": {
      "text_english_vector": [0.0123, -0.0456, ...],
      "text_arabic_vector": [-0.0789, 0.0321, ...]
    }
  }
  ```

### Step 9: Store Both Vectors in One Qdrant Point
- **Collection**: `financial_documents`.
- **Collection Vector Configuration**:
  ```python
  vectors_config = {
      "text_english_vector": models.VectorParams(
          size=validated_english_dim,
          distance=models.Distance.COSINE
      ),
      "text_arabic_vector": models.VectorParams(
          size=validated_arabic_dim,
          distance=models.Distance.COSINE
      )
  }
  ```
- **Payload Schema**:
  - `chunk_id`, `document_id`, `element_ids`, `chunk_index`, `content_type`
  - `original_language`, `original_text`, `text_english`, `text_arabic`
  - `translation_status`, `source`, `embedding_models`
  - **Rules**: Embeddings are stored only in Qdrant named vector fields, never in payload. Private file storage paths and API keys are never stored.
- **Execution**:
  - Upsert point into Qdrant using `qdrant_point_id`.
  - Update PostgreSQL `document_chunks.status = 'PROCESSED'`.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "chunk_id": "018f3a2c-...",
    "qdrant_point_id": "018f3a2c-...",
    "collection": "financial_documents",
    "status": "STORED",
    "vectors_stored": ["text_english_vector", "text_arabic_vector"]
  }
  ```

### Step 10: Deep Qdrant Storage Verification
- **Process**:
  - Fetch point from Qdrant by `qdrant_point_id` (`with_vectors=True, with_payload=True`).
  - Perform 8-point verification:
    1. Point exists in `financial_documents`.
    2. Both named vectors (`text_english_vector`, `text_arabic_vector`) are present.
    3. `len(text_english_vector) == validated_english_dim`.
    4. `len(text_arabic_vector) == validated_arabic_dim`.
    5. Vector values are all finite numbers (no `NaN`, `Inf`).
    6. Payload `chunk_id` and `document_id` match PostgreSQL manifest.
    7. Payload contains valid `original_text`, `text_english`, and `text_arabic`.
    8. Payload model identifiers and revisions match the ingestion configuration.
  - If all checks pass: mark chunk in `document_chunks` as `status = 'VERIFIED'`.
  - Increment `ingestion_jobs.verified_chunks`.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "expected_chunks": 2,
    "verified_chunks": 2,
    "verified_english_vectors": 2,
    "verified_arabic_vectors": 2,
    "status": "VERIFIED"
  }
  ```

### Step 11: Update PostgreSQL & Complete Ingestion
- **Logic**:
  - Query manifest: `total_expected = count(document_chunks)`, `total_verified = count(document_chunks where status = 'VERIFIED')`.
  - **Success Path**:
    - If `total_verified == total_expected` and `total_expected > 0`:
      - Set `documents.status = 'ACTIVE'`, `total_chunks = total_verified`.
      - Set `ingestion_jobs.status = 'COMPLETED'`, `current_step = 'COMPLETED'`, `completed_at = now()`.
  - **Failure Path**:
    - If any chunk fails or `total_verified < total_expected`:
      - Set `documents.status = 'FAILED'`, `failure_reason = ...`.
      - Set `ingestion_jobs.status = 'FAILED'`, `error_step = 'QDRANT_VERIFICATION'`.
  - Retries increment `attempt_number` and preserve previous job attempts for audit.
- **Output**:
  ```json
  {
    "document_id": "018f3a2c-...",
    "job_id": "018f3a2c-...",
    "document_status": "ACTIVE",
    "job_status": "COMPLETED"
  }
  ```

### Step 12: Ingestion Health & Reconciliation
- **Periodic / CLI Reconciler**:
  - Detects manifest chunks missing from Qdrant.
  - Detects Qdrant points missing either named vector or required payload fields.
  - Detects dimension mismatches against active model configuration.
  - Detects orphaned points in Qdrant (points referencing nonexistent or deleted documents).
  - Detects documents marked `ACTIVE` without 100% verified manifest chunks.
  - Detects jobs stuck in `PROCESSING` beyond timeout threshold (e.g., 30 minutes).
- **Repair Actions**:
  - Mark stuck jobs as `FAILED` with retry option.
  - Re-enqueue unverified or missing chunks idempotently.
  - Report / safely quarantine orphan vectors with audit logs.
  - CLI usage: `python -m src.reconciliation.cli --audit` and `python -m src.reconciliation.cli --repair`.

---

## 6. Project Directory Structure

```
d:/rag-assistant-v1/
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── alembic.ini
├── private_storage/
├── alembic/
│   ├── env.py
│   └── versions/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── session.py
│   │   └── models.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── private_store.py
│   ├── queue/
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   └── tasks.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── pdf_parser.py
│   │   ├── ocr_parser.py
│   │   ├── docx_parser.py
│   │   ├── xlsx_parser.py
│   │   └── csv_parser.py
│   ├── normalizers/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   └── normalizer.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── adaptive_chunker.py
│   ├── language/
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── translation/
│   │   ├── __init__.py
│   │   └── gpt4o_service.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── english_service.py
│   │   ├── arabic_service.py
│   │   └── validator.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py
│   │   └── collection.py
│   ├── verification/
│   │   ├── __init__.py
│   │   └── verifier.py
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── reconciliation/
│   │   ├── __init__.py
│   │   ├── reconciler.py
│   │   └── cli.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py
│   │   │   └── schemas.py
│   │   └── errors.py
│   └── main.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── test_step1_upload.py
    ├── test_step2_routing.py
    ├── test_step3_extraction.py
    ├── test_step4_normalization.py
    ├── test_step5_chunking.py
    ├── test_step6_language.py
    ├── test_step7_translation.py
    ├── test_step8_embeddings.py
    ├── test_step9_qdrant_storage.py
    ├── test_step10_verification.py
    ├── test_step11_orchestration.py
    └── test_step12_reconciliation.py
```

---

## 7. Environment Variables (`.env.example`)

```dotenv
APP_ENV=development
APP_NAME=financial-rag
API_V1_PREFIX=/api/v1

# PostgreSQL Database
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/financial_rag

# Qdrant Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=financial_documents

# OpenAI Translation
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_TRANSLATION_MODEL=gpt-4o

# Separate English Sentence Transformers Model
ENGLISH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENGLISH_EMBEDDING_REVISION=
ENGLISH_EMBEDDING_DIMENSION=

# Separate Arabic Sentence Transformers Model
ARABIC_EMBEDDING_MODEL=Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2
ARABIC_EMBEDDING_REVISION=
ARABIC_EMBEDDING_DIMENSION=

# Azure AI Document Intelligence (OCR for scanned documents)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=

# Redis & Celery Background Queue
REDIS_URL=redis://localhost:6379/0

# Private Deterministic Storage
PRIVATE_STORAGE_ROOT=./private_storage
MAX_UPLOAD_SIZE_MB=50

# Adaptive Chunking Constraints
CHUNK_SIZE_TOKENS=384
CHUNK_OVERLAP_TOKENS=48
PROCESSING_TIMEOUT_MINUTES=30
```

---

## 8. Phased Implementation Roadmap

| Phase | Steps Covered | Scope & Key Deliverables | Verification Gate |
|---|---|---|---|
| **Phase 1** | Scaffolding & Setup | Docker Compose, `pyproject.toml`, `.env.example`, settings, SQLAlchemy models, Alembic migrations. | Containers healthy; DB schema created with all constraints and indexes. |
| **Phase 2** | Step 1 & Step 2 | Private storage manager, `POST /api/v1/documents`, PyMuPDF inspection & file router service. | Valid uploads return HTTP 202; files stored privately; corrupt/encrypted files rejected. |
| **Phase 3** | Step 3 & Step 4 | PyMuPDF, Azure OCR client, python-docx, openpyxl, and CSV parsers; `NormalizedElement` conversion. | Financial tables, currencies, dates, and source references extracted and typed accurately. |
| **Phase 4** | Step 5 | Tokenizers, adaptive table splitter with header repetition, deterministic manifest generator, `document_chunks` table persistence. | Chunks respect token boundaries; manifest persisted in DB with stable UUIDs. |
| **Phase 5** | Step 6 & Step 7 | Lingua detector service; OpenAI GPT-4o translation service with financial entity preservation safeguards. | Language detected (`en`/`ar`); bilingual pairs produced without altering numbers or facts. |
| **Phase 6** | Step 8 | `EnglishEmbeddingService`, `ArabicEmbeddingService` (`Arabic-Triplet-Matryoshka-V2`), startup validator. | Dynamic dimension validation passes; no NaN/Inf vectors; correct model pooling applied. |
| **Phase 7** | Step 9 & Step 10 | Qdrant collection setup (`financial_documents`), dual named vector point upsert, 8-point verification service. | Points stored with both named vectors; verification passes all checks and sets chunks `VERIFIED`. |
| **Phase 8** | Step 11 | Celery worker, end-to-end 12-step pipeline orchestrator, document activation, attempt tracking. | Full document moves to `ACTIVE`; jobs complete with accurate chunk counters. |
| **Phase 9** | Step 12 | Periodic / CLI reconciliation engine (`--audit`, `--repair`), orphan cleaner, stuck-job recovery. | Anomaly detection correctly identifies and repairs drifted points and stuck jobs. |
| **Phase 10** | Comprehensive Testing | Pytest unit, mock, and integration test suite across PDF, DOCX, XLSX, and CSV; test coverage report. | All test suites pass 100%. |
