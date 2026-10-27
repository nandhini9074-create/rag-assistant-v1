import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from backend.config import get_settings
from backend.search.api.schemas import Citation
from backend.search.context.builder import ContextChunk

logger = logging.getLogger("search-generation")


@dataclass
class DraftAnswerResult:
    """
    Step 9: Answer Generation Output Schema.
    """
    request_id: str
    answer: str
    response_language: str
    citations: list[Citation] = field(default_factory=list)
    generation_status: str = "COMPLETED"
    model_used: str = "none"


class AnswerGenerator:
    """
    Step 9: Evidence-Grounded Answer Generation.
    - Produces draft answers using only verified context from Step 8.
    - Grounds answers strictly in the user's requested response language (ar / en).
    - Uses GPT-4o as primary generator with automatic Gemini 3.5 Flash Lite fallback.
    - Enforces citation references [Chunk CH-XXX].
    - Handles insufficient evidence gracefully without hallucinating numbers.
    """

    SYSTEM_PROMPT = """You are an evidence-grounded financial assistant specialized in Arabic and English corporate reports.
Your task is to generate a clear, accurate answer using ONLY the provided evidence context.

CRITICAL INSTRUCTIONS:
1. USE PROVIDED EVIDENCE ONLY: Do not assume or extrapolate facts not in the text.
2. PRESERVE NUMBERS & CURRENCIES: Never alter numbers, currencies (AED, USD, SAR), signs, or dates.
3. PREVENT FALSE AGGREGATION: If a chunk displays an individual transaction or partial line item, do NOT claim it represents the entire monthly or annual total unless explicitly verified by the document text.
4. MANDATORY CITATIONS: Cite every claim using the chunk marker, e.g., [Chunk CH-001].
5. INSUFFICIENT EVIDENCE: If the evidence does not contain the answer, explicitly state that the documents do not provide this information.
6. LANGUAGE: Respond strictly in the designated response language (Arabic for 'ar', English for 'en').
"""

    def __init__(self) -> None:
        settings = get_settings()
        self.app_env = settings.APP_ENV
        self.openai_api_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_TRANSLATION_MODEL or "gpt-4o"
        self.gemini_api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        self.gemini_model = settings.GEMINI_TRANSLATION_MODEL or "gemini-3.5-flash-lite"

        # Initialize OpenAI client
        if self.openai_api_key and self.openai_api_key not in ("test_openai_key", ""):
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            self.openai_client = None

        # Initialize Gemini client
        if self.gemini_api_key and self.gemini_api_key not in ("test_gemini_key", ""):
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            self.gemini_client = None

    async def _call_openai(self, prompt: str, system_prompt: str) -> str:
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured")

        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not configured")

        combined_prompt = f"{system_prompt}\n\nUser Question and Evidence:\n{prompt}"
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=self.gemini_model,
            contents=combined_prompt,
        )
        return getattr(response, "text", "") or ""

    def _extract_citations(
        self,
        answer_text: str,
        context_chunks: list[ContextChunk],
    ) -> list[Citation]:
        """
        Extract cited chunk identifiers from answer text (e.g. [Chunk CH-001])
        and match with corresponding context chunks.
        """
        cited_tags = set(re.findall(r"\[Chunk\s+([^\]]+)\]", answer_text, re.IGNORECASE))
        citations: list[Citation] = []
        matched_chunks: set[str] = set()

        for chunk in context_chunks:
            tag = f"CH-{chunk.relevance_rank:03d}"
            # Match by tag or raw chunk_id
            if tag in cited_tags or chunk.chunk_id in cited_tags:
                matched_chunks.add(chunk.chunk_id)
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        file_name=chunk.source.get("file_name", "unknown"),
                        page=chunk.source.get("page"),
                        table_index=chunk.source.get("table_index"),
                    )
                )

        # Fallback: If model did not emit formatted bracket tags, cite top referenced context chunks
        if not citations and context_chunks:
            for chunk in context_chunks[:3]:
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        file_name=chunk.source.get("file_name", "unknown"),
                        page=chunk.source.get("page"),
                        table_index=chunk.source.get("table_index"),
                    )
                )

        return citations

    async def generate_draft_answer(
        self,
        request_id: str,
        original_query: str,
        response_language: str,
        formatted_context_str: str,
        context_chunks: list[ContextChunk],
    ) -> DraftAnswerResult:
        """
        Generate draft evidence-grounded answer using LLM with multi-tier fallback.
        """
        if not context_chunks or not formatted_context_str.strip():
            logger.info(f"[{request_id}] Zero evidence chunks available for answer generation.")
            insufficient_msg = (
                "لم يتم العثور على معلومات كافية في المستندات المعتمدة للإجابة على هذا السؤال."
                if response_language == "ar"
                else "Insufficient evidence available in the authorized documents to answer this question."
            )
            return DraftAnswerResult(
                request_id=request_id,
                answer=insufficient_msg,
                response_language=response_language,
                citations=[],
                generation_status="INSUFFICIENT_EVIDENCE",
                model_used="rule_fallback",
            )

        # In test mode, produce deterministic mock grounded answer
        if self.app_env == "test":
            top_chunk = context_chunks[0]
            snippet = top_chunk.text_for_llm.strip().replace("\n", " ")[:160]
            if response_language == "ar":
                mock_answer = (
                    f"بناءً على الأدلة المسترجعة، يُظهر السجل المالي: {snippet} [Chunk CH-001]."
                )
            else:
                mock_answer = (
                    f"Based on the authorized financial documents: {snippet} [Chunk CH-001]."
                )
            return DraftAnswerResult(
                request_id=request_id,
                answer=mock_answer,
                response_language=response_language,
                citations=self._extract_citations(mock_answer, context_chunks),
                generation_status="COMPLETED",
                model_used="mock_test",
            )

        user_prompt = (
            f"Query: {original_query}\n"
            f"Response Language: {response_language}\n\n"
            f"{formatted_context_str}"
        )

        model_used = "gpt-4o"
        raw_answer = ""

        # Tier 1: GPT-4o
        if self.openai_client:
            try:
                logger.info(f"[{request_id}] Calling OpenAI ({self.openai_model}) for answer generation...")
                raw_answer = await self._call_openai(user_prompt, self.SYSTEM_PROMPT)
            except Exception as e:
                logger.warning(f"[{request_id}] OpenAI answer generation failed: {e}. Falling back to Gemini.")
                model_used = self.gemini_model

        # Tier 2: Gemini fallback
        if not raw_answer and self.gemini_client:
            try:
                logger.info(f"[{request_id}] Calling Gemini ({self.gemini_model}) for answer generation...")
                raw_answer = await self._call_gemini(user_prompt, self.SYSTEM_PROMPT)
                model_used = self.gemini_model
            except Exception as e:
                logger.error(f"[{request_id}] Gemini answer generation also failed: {e}")

        # Tier 3: Deterministic fallback if API keys absent or network fails
        if not raw_answer:
            logger.warning(f"[{request_id}] LLM generation unavailable. Generating extractive context summary.")
            top_chunk = context_chunks[0]
            snippet = top_chunk.text_for_llm.strip().replace("\n", " ")[:200]
            if response_language == "ar":
                raw_answer = f"تُشير المستندات المتاحة إلى: {snippet} [Chunk CH-001]."
            else:
                raw_answer = f"The available financial records indicate: {snippet} [Chunk CH-001]."
            model_used = "extractive_fallback"

        citations = self._extract_citations(raw_answer, context_chunks)
        logger.info(f"[{request_id}] Answer generated using {model_used} with {len(citations)} citation(s).")

        return DraftAnswerResult(
            request_id=request_id,
            answer=raw_answer,
            response_language=response_language,
            citations=citations,
            generation_status="COMPLETED",
            model_used=model_used,
        )
