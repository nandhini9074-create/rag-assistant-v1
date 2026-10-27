# Multilingual Financial RAG Ingestion Pipeline

A production-ready document ingestion pipeline for a multilingual financial RAG system, implementing strict two-language Sentence Transformers embedding generation (English and Arabic), deep vector verification in Qdrant, and comprehensive reconciliation.

## Key Features
- **12 Sequential Ingestion Steps** with typed Pydantic contracts and fault isolation.
- **Separate Sentence Transformers Models**:
  - English: Configurable (`ENGLISH_EMBEDDING_MODEL`), e.g. `sentence-transformers/all-MiniLM-L6-v2`.
  - Arabic: `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` with pinned revision support.
  - Strictly **no OpenAI embeddings**.
  - Dynamic runtime model dimension validation without hardcoding.
- **Dual Named Vectors in One Qdrant Point**:
  - `text_english_vector` and `text_arabic_vector` in collection `financial_documents`.
  - Vectors stored only in vector indices, never in payload.
  - Private file paths and secrets strictly excluded from payloads.
- **Multilingual Support**:
  - PDF (PyMuPDF text or Azure AI Document Intelligence OCR for scans).
  - DOCX, XLSX, and CSV with preservation of financial signs, currencies, and decimal precision.
- **Language Detection & Bilingual Translation**:
  - Lingua language detection (`en` and `ar`).
  - OpenAI GPT-4o translation with financial entity safeguards (numbers, currencies, dates).
- **PostgreSQL Source of Truth**:
  - Tables: `documents`, `ingestion_jobs`, `document_chunks`.
  - Full audit trail and retry idempotency.
- **Health & Reconciliation Engine**:
  - Automated detection of drift, missing points, dimension mismatches, orphan vectors, and stuck jobs.

---

## Getting Started

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose (optional for local PostgreSQL, Redis, and Qdrant)

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in necessary keys:
```bash
cp .env.example .env
```

### 3. Start Local Services (Docker)
```bash
docker compose up -d
```

### 4. Run Database Migrations
```bash
alembic upgrade head
```

### 5. Run the Ingestion API
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

### Step 1: Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@financial_report.pdf"
```

**Response (HTTP 202 Accepted):**
```json
{
  "document_id": "018f3a2c-1234-7000-8000-000000000001",
  "job_id": "018f3a2c-1234-7000-8000-000000000002",
  "file_name": "financial_report.pdf",
  "file_type": "pdf",
  "status": "UPLOADED",
  "job_status": "QUEUED",
  "created_at": "2026-10-27T10:00:00Z"
}
```

### Check Ingestion Status
```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}"
```

### Retrieve Ingestion Chunk Manifest
```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}/chunks"
```

---

## Health Audit and Reconciliation (Step 12)

Run audit without changes:
```bash
python -m src.reconciliation.cli --audit
```

Run audit and apply automated repairs:
```bash
python -m src.reconciliation.cli --repair
```

---

## Running Tests
```bash
pytest
```
