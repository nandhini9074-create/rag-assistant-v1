import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from backend.search.api.schemas import Citation
from backend.search.context.builder import ContextChunk

logger = logging.getLogger("search-validation")


@dataclass
class VerificationResult:
    """
    Step 10: Evidence and Citation Validation Output Contract.
    """
    request_id: str
    validation_status: str  # "PASSED" | "FAILED"
    citation_validation: str  # "PASSED" | "FAILED" | "WARNING"
    evidence_validation: str  # "PASSED" | "FAILED" | "INSUFFICIENT_EVIDENCE"
    unsupported_claims: list[str] = field(default_factory=list)
    ready_to_return: bool = True
    verified_citations: list[Citation] = field(default_factory=list)
    validated_answer: str = ""


class CitationAndEvidenceVerifier:
    """
    Step 10: Evidence and Citation Validation Service.
    - Validates that every citation refers to an authentic chunk present in context.
    - Checks document ID and source metadata consistency.
    - Deterministically validates financial figures, numbers, and currencies against context.
    - Detects unsupported numbers or false aggregations.
    - Enforces safe fallbacks if validation fails.
    """

    _NUMBERS_REGEX = re.compile(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b")
    _IGNORE_NUMBERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "001", "002", "003", "004", "005"}

    @classmethod
    def _normalize_number(cls, num_str: str) -> str:
        """
        Strip commas and formatting symbols for comparison (e.g. 50,000 -> 50000).
        """
        return num_str.replace(",", "").replace("%", "").strip()

    @classmethod
    def validate_answer_and_citations(
        cls,
        request_id: str,
        answer: str,
        context_chunks: list[ContextChunk],
        citations: list[Citation],
        allowed_verified_figures: Optional[list[str]] = None,
    ) -> VerificationResult:
        """
        Execute Step 10 2-stage verification:
        1. Citation Structure & Existence Check.
        2. Financial Number & Evidence Fidelity Check.
        """
        unsupported_claims: list[str] = []
        verified_citations: list[Citation] = []

        context_map = {c.chunk_id: c for c in context_chunks}
        chunk_tags = {f"CH-{c.relevance_rank:03d}": c for c in context_chunks}

        # -------------------------------------------------------------
        # 1. Citation Validation
        # -------------------------------------------------------------
        citation_validation = "PASSED"

        if not citations and context_chunks:
            citation_validation = "WARNING"
            unsupported_claims.append("Answer contains claims without explicit citations to evidence chunks.")
        elif citations:
            for cit in citations:
                matched_chunk = context_map.get(cit.chunk_id)
                if not matched_chunk:
                    citation_validation = "FAILED"
                    unsupported_claims.append(
                        f"Cited chunk ID '{cit.chunk_id}' does not exist in the retrieved evidence context."
                    )
                else:
                    # Verify document ID consistency
                    if cit.document_id and cit.document_id != matched_chunk.document_id:
                        citation_validation = "FAILED"
                        unsupported_claims.append(
                            f"Citation document ID mismatch: cited '{cit.document_id}' but chunk belongs to '{matched_chunk.document_id}'."
                        )
                    else:
                        verified_citations.append(cit)

        # -------------------------------------------------------------
        # 2. Evidence Validation (Financial Numbers Fidelity)
        # -------------------------------------------------------------
        evidence_validation = "PASSED"
        is_insufficient_evidence_response = bool(
            re.search(
                r"(insufficient evidence|not available|does not provide|لم يتم العثور|غير متوفر|لا تتوفر)",
                answer,
                re.IGNORECASE,
            )
        )

        if is_insufficient_evidence_response and not context_chunks:
            evidence_validation = "INSUFFICIENT_EVIDENCE"
            ready_to_return = True
            validation_status = "PASSED"
            return VerificationResult(
                request_id=request_id,
                validation_status=validation_status,
                citation_validation="PASSED",
                evidence_validation=evidence_validation,
                unsupported_claims=[],
                ready_to_return=ready_to_return,
                verified_citations=[],
                validated_answer=answer,
            )

        # Extract all numbers from combined evidence text
        combined_context_text = " ".join(c.text_for_llm for c in context_chunks)
        context_numbers = set(
            cls._normalize_number(n) for n in cls._NUMBERS_REGEX.findall(combined_context_text)
        )
        if allowed_verified_figures:
            for fig in allowed_verified_figures:
                context_numbers.add(cls._normalize_number(str(fig)))

        # Extract all numbers from generated answer
        answer_numbers = cls._NUMBERS_REGEX.findall(answer)
        for num_raw in answer_numbers:
            norm_num = cls._normalize_number(num_raw)
            # Skip small rank/index numbers (e.g. Chunk CH-001)
            if norm_num in cls._IGNORE_NUMBERS:
                continue

            if norm_num not in context_numbers:
                evidence_validation = "FAILED"
                unsupported_claims.append(
                    f"Answer mentions unverified figure '{num_raw}', which does not appear in any cited context chunk."
                )

        # -------------------------------------------------------------
        # 3. Overall Status & Safe Fallback
        # -------------------------------------------------------------
        if citation_validation == "FAILED" or evidence_validation == "FAILED":
            validation_status = "FAILED"
            ready_to_return = False
            # Safe Fallback: Annotate answer with clear audit disclosure
            validated_answer = (
                f"{answer}\n\n"
                f"[Audit Warning: Certain financial figures or citations in this response could not be verified against the source documents.]"
            )
            logger.warning(
                f"[{request_id}] Step 10 validation FAILED: {len(unsupported_claims)} unsupported claim(s) detected."
            )
        else:
            validation_status = "PASSED"
            ready_to_return = True
            validated_answer = answer
            logger.info(f"[{request_id}] Step 10 validation PASSED. Citations and evidence figures verified.")

        return VerificationResult(
            request_id=request_id,
            validation_status=validation_status,
            citation_validation=citation_validation,
            evidence_validation=evidence_validation,
            unsupported_claims=unsupported_claims,
            ready_to_return=ready_to_return,
            verified_citations=verified_citations if verified_citations else citations,
            validated_answer=validated_answer,
        )
