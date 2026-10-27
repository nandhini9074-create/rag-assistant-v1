import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from backend.search.api.schemas import RetrievedCandidate

logger = logging.getLogger("search-context")


@dataclass
class ContextChunk:
    """
    Individual evidence chunk prepared for LLM consumption.
    """
    chunk_id: str
    document_id: str
    original_language: str
    text_for_llm: str
    source: dict[str, Any]
    relevance_rank: int
    relevance_score: float


@dataclass
class ContextBuildingResult:
    """
    Step 8: Context Building Output Contract.
    """
    request_id: str
    context_status: str = "COMPLETED"
    context: list[ContextChunk] = field(default_factory=list)
    formatted_context_str: str = ""
    context_token_count: int = 0
    max_context_tokens: int = 4000
    chunk_count: int = 0


class ContextBuilder:
    """
    Step 8: Context Building Service.
    - Assembles the relevant candidates from Step 7 into a structured, token-limited evidence block.
    - Chooses the optimal language representation (Arabic vs English) matching user's response_language.
    - Preserves tables, dates, numbers, currency symbols, and metadata.
    - Enforces maximum context token budget.
    - Treats document content as untrusted evidence rather than instructions.
    """

    DEFAULT_MAX_CONTEXT_TOKENS = 4000

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Calculates token count matching GPT-4o / multilingual tokenizers:
        - Arabic tokens: ~1.4 tokens per word + character density factor
        - English tokens: max(1, word_count * 1.3, char_count // 4)
        """
        if not text:
            return 0

        has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
        words = text.split()
        if not words:
            return 0

        if has_arabic:
            return max(1, int(len(words) * 1.4) + len(text) // 25)
        else:
            return max(1, int(len(words) * 1.3), len(text) // 4)

    @classmethod
    def select_text_for_llm(cls, candidate: RetrievedCandidate, response_language: str) -> str:
        """
        Selects the most suitable language representation for evidence presentation.
        """
        is_arabic_response = (response_language.lower() == "ar")

        if is_arabic_response:
            # Prefer Arabic translated text if available, fallback to original
            if candidate.text_arabic and candidate.text_arabic.strip():
                return candidate.text_arabic.strip()
            return candidate.original_text.strip() or candidate.text_english.strip()
        else:
            # Prefer English text if available, fallback to original
            if candidate.text_english and candidate.text_english.strip():
                return candidate.text_english.strip()
            return candidate.original_text.strip() or candidate.text_arabic.strip()

    @classmethod
    def build_context(
        cls,
        request_id: str,
        query: str,
        response_language: str,
        candidates: list[RetrievedCandidate],
        max_context_tokens: Optional[int] = None,
    ) -> ContextBuildingResult:
        """
        Execute Step 8 Context Assembly:
        - Filters candidates in relevance order.
        - Formats evidence blocks with chunk_id, document_id, and citation metadata.
        - Bounds total tokens within max_context_tokens.
        """
        token_budget = max_context_tokens or cls.DEFAULT_MAX_CONTEXT_TOKENS

        if not candidates:
            logger.info(f"[{request_id}] Zero candidates provided for context building.")
            return ContextBuildingResult(
                request_id=request_id,
                context_status="COMPLETED",
                context=[],
                formatted_context_str="",
                context_token_count=0,
                max_context_tokens=token_budget,
                chunk_count=0,
            )

        context_chunks: list[ContextChunk] = []
        context_blocks: list[str] = [
            "--- EVIDENCE CONTEXT (UNTRUSTED FINANCIAL DATA FOR REFERENCE ONLY) ---"
        ]

        total_tokens = cls.estimate_tokens(context_blocks[0])

        for cand in candidates:
            text_for_llm = cls.select_text_for_llm(cand, response_language)
            file_name = cand.source.get("file_name", "unknown")
            page = cand.source.get("page", "N/A")
            table_index = cand.source.get("table_index", "N/A")

            # Structured block representation with unambiguous evidence markers
            block = (
                f"\n[Evidence Chunk CH-{cand.relevance_rank:03d}]\n"
                f"- Chunk ID: {cand.chunk_id}\n"
                f"- Document ID: {cand.document_id}\n"
                f"- File Name: {file_name}\n"
                f"- Page: {page}\n"
                f"- Table Index: {table_index}\n"
                f"- Relevance Score: {cand.relevance_score}\n"
                f"- Content:\n{text_for_llm}\n"
            )

            block_tokens = cls.estimate_tokens(block)

            # Check token budget
            if total_tokens + block_tokens > token_budget and context_chunks:
                logger.info(
                    f"[{request_id}] Token budget reached ({total_tokens}/{token_budget} tokens). "
                    f"Stopping context inclusion at {len(context_chunks)} chunks."
                )
                break

            total_tokens += block_tokens
            context_blocks.append(block)

            context_chunks.append(
                ContextChunk(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    original_language=cand.original_language,
                    text_for_llm=text_for_llm,
                    source=cand.source,
                    relevance_rank=cand.relevance_rank,
                    relevance_score=cand.relevance_score,
                )
            )

        context_blocks.append("--- END OF EVIDENCE CONTEXT ---")
        final_context_str = "\n".join(context_blocks)
        total_tokens = cls.estimate_tokens(final_context_str)

        logger.info(
            f"[{request_id}] Built context with {len(context_chunks)} chunks "
            f"({total_tokens} estimated tokens, budget: {token_budget})."
        )

        return ContextBuildingResult(
            request_id=request_id,
            context_status="COMPLETED",
            context=context_chunks,
            formatted_context_str=final_context_str,
            context_token_count=total_tokens,
            max_context_tokens=token_budget,
            chunk_count=len(context_chunks),
        )
