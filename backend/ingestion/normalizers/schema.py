import uuid
from typing import Any, Optional
from pydantic import BaseModel, Field


class ExtractedItem(BaseModel):
    """Raw item extracted by a parser in Step 3."""
    type: str  # 'text', 'table', 'heading'
    text: Optional[str] = None
    headers: Optional[list[str]] = None
    rows: Optional[list[list[str]]] = None
    source: dict[str, Any] = Field(default_factory=dict)


class NormalizedElement(BaseModel):
    """Common typed representation in Step 4."""
    element_id: uuid.UUID
    content_type: str  # 'financial_table', 'narrative_text', 'heading', 'transaction_record'
    text: str
    source: dict[str, Any] = Field(default_factory=dict)
    raw_data: Optional[dict[str, Any]] = None


class LogicalChunk(BaseModel):
    """Chunk model produced in Step 5."""
    chunk_id: uuid.UUID
    chunk_index: int
    element_ids: list[uuid.UUID]
    content_type: str
    text: str
    source: dict[str, Any] = Field(default_factory=dict)
    qdrant_point_id: uuid.UUID
