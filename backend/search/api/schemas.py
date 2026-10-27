import uuid
from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Step 1 & 2 Request Schema:
    Receives user query, optional document names scope, and conversation context.
    """
    query: str = Field(..., min_length=1, max_length=2000, description="Financial question in Arabic or English")
    document_names: Optional[list[str]] = Field(None, description="Optional list of specific filenames to scope search")
    conversation_id: Optional[str] = Field(None, description="Optional conversation tracking identifier")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of vector candidates to retrieve")


class Citation(BaseModel):
    """
    Step 11 Citation item.
    """
    chunk_id: str
    document_id: str
    file_name: str
    page: Optional[int] = None
    table_index: Optional[int] = None


class AnswerMetadata(BaseModel):
    """
    Step 10 & 11 Quality and Verification metadata.
    """
    citation_validation: str = Field("PASSED", description="PASSED | FAILED | WARNING")
    evidence_validation: str = Field("PASSED", description="PASSED | FAILED | INSUFFICIENT_EVIDENCE")
    calculation_performed: bool = Field(False, description="True if deterministic arithmetic calculation was performed")
    unsupported_claims: list[str] = Field(default_factory=list, description="List of unverified claims detected during audit")


class QueryResponse(BaseModel):
    """
    Step 11 Final API Response.
    """
    request_id: str = Field(..., description="Unique backend-generated tracking ID (REQ-xxxx)")
    conversation_id: Optional[str] = None
    status: str = Field("COMPLETED", description="COMPLETED | INSUFFICIENT_EVIDENCE | FAILED")
    answer: str
    response_language: str = Field(..., description="'ar' or 'en'")
    citations: list[Citation] = Field(default_factory=list)
    answer_metadata: AnswerMetadata


class RetrievedCandidate(BaseModel):
    """
    Internal intermediate structure for retrieved chunks before context building.
    """
    chunk_id: str
    document_id: str
    retrieval_score: float
    relevance_rank: int = 1
    relevance_score: float = 0.0
    text_english: str
    text_arabic: str
    original_text: str
    original_language: str
    source: dict[str, Any] = Field(default_factory=dict)
