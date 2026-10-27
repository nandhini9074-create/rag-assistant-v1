import uuid
from typing import Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.constants import DocumentStatus
from backend.database.models import Document


class ValidationResult(BaseModel):
    """
    Step 2 Output Data Contract: Validation passed.
    """
    request_id: str
    validation_status: str = "PASSED"
    query: str
    authorized_document_ids: list[uuid.UUID] = Field(default_factory=list)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DocumentAccessResolver:
    """
    Step 2: Request Validation and Access Control.
    Resolves filenames to verified, active document IDs in PostgreSQL.
    Enforces authorization scope so users never search unauthorized documents.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def validate_and_resolve(
        self,
        request_id: str,
        query: str,
        document_names: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Executes Step 2:
        1. Validates query non-emptiness and length.
        2. Resolves selected document names against PostgreSQL.
        3. Enforces authorization, active status, and disambiguation.
        4. Validates conversation ownership.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Query cannot be empty or whitespace.",
            )
        if len(cleaned_query) > 2000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Query exceeds maximum allowed length of 2000 characters.",
            )

        if document_names is not None and len(document_names) > 0:
            authorized_ids = await self._resolve_specific_documents(document_names, user_id)
        else:
            authorized_ids = await self._resolve_all_authorized_documents(user_id)

        # Validate conversation_id format if provided
        if conversation_id and not conversation_id.strip():
            conversation_id = None

        return ValidationResult(
            request_id=request_id,
            validation_status="PASSED",
            query=cleaned_query,
            authorized_document_ids=authorized_ids,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def _resolve_specific_documents(
        self,
        document_names: list[str],
        user_id: Optional[str] = None,
    ) -> list[uuid.UUID]:
        # Clean and validate structure of document_names
        cleaned_names = [name.strip() for name in document_names if name and name.strip()]
        if not cleaned_names:
            return await self._resolve_all_authorized_documents(user_id)

        resolved_ids: list[uuid.UUID] = []

        for name in cleaned_names:
            # Query all documents matching filename ordered by created_at DESC for disambiguation
            stmt = (
                select(Document)
                .where(Document.file_name == name)
                .order_by(desc(Document.created_at))
            )
            matches = (await self.db.execute(stmt)).scalars().all()

            if not matches:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Selected document '{name}' was not found in the document registry.",
                )

            # Check user ownership if user_id is provided and multitenancy is active
            if user_id:
                user_matches = [m for m in matches if m.uploaded_by == user_id or m.uploaded_by is None]
                if not user_matches:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied: You are not authorized to view document '{name}'.",
                    )
                target_doc = user_matches[0]
            else:
                target_doc = matches[0]

            # Verify document is in ACTIVE status
            if target_doc.status != DocumentStatus.ACTIVE.value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Selected document '{name}' is not ready for retrieval "
                        f"(current status: {target_doc.status}). Only ACTIVE documents can be searched."
                    ),
                )

            resolved_ids.append(target_doc.document_id)

        return resolved_ids

    async def _resolve_all_authorized_documents(
        self,
        user_id: Optional[str] = None,
    ) -> list[uuid.UUID]:
        stmt = select(Document.document_id).where(Document.status == DocumentStatus.ACTIVE.value)
        if user_id:
            stmt = stmt.where((Document.uploaded_by == user_id) | (Document.uploaded_by.is_(None)))

        doc_ids = (await self.db.execute(stmt)).scalars().all()
        return list(doc_ids)
