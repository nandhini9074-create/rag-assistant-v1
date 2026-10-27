from enum import Enum


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class ContentType(str, Enum):
    FINANCIAL_TABLE = "financial_table"
    NARRATIVE_TEXT = "narrative_text"
    HEADING = "heading"
    TRANSACTION_RECORD = "transaction_record"
    RAW_TEXT = "raw_text"


class SupportedFileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"


class ParserType(str, Enum):
    PYMUPDF = "pymupdf"
    AZURE_OCR = "azure_ocr"
    PYTHON_DOCX = "python-docx"
    OPENPYXL = "openpyxl"
    CSV = "csv"


class TranslationStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class PipelineStep(str, Enum):
    FILE_VALIDATION = "FILE_VALIDATION"
    FILE_ROUTING = "FILE_ROUTING"
    CONTENT_EXTRACTION = "CONTENT_EXTRACTION"
    DATA_NORMALIZATION = "DATA_NORMALIZATION"
    ADAPTIVE_CHUNKING = "ADAPTIVE_CHUNKING"
    LANGUAGE_DETECTION = "LANGUAGE_DETECTION"
    TRANSLATION = "TRANSLATION"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"
    QDRANT_STORAGE = "QDRANT_STORAGE"
    QDRANT_VERIFICATION = "QDRANT_VERIFICATION"
    PIPELINE_COMPLETION = "PIPELINE_COMPLETION"
    COMPLETED = "COMPLETED"
