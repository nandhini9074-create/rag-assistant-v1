import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from backend.search.api.schemas import RetrievedCandidate

logger = logging.getLogger("search-reranking")


@dataclass
class RerankingResult:
    """
    Step 7: Result Reranking and Relevance Filtering Output Contract.
    """
    request_id: str
    reranking_status: str = "COMPLETED"
    results: list[RetrievedCandidate] = field(default_factory=list)
    result_count: int = 0
    filtered_out_count: int = 0


class ResultReranker:
    """
    Step 7: Result Reranking and Relevance Filtering Service.
    - Improves candidate ordering by scoring semantic retrieval similarity combined
      with exact financial entity and token overlap.
    - Keeps original retrieval_score separate from relevance_score.
    - Deduplicates identical chunks.
    - Applies a minimum relevance threshold to filter out noise.
    """

    DEFAULT_MIN_THRESHOLD = 0.35

    @classmethod
    def _calculate_token_overlap_score(
        cls,
        candidate_text: str,
        query: str,
        financial_tokens: list[str],
    ) -> float:
        """
        Calculate a normalized lexical & financial entity overlap score between 0.0 and 1.0.
        """
        if not candidate_text:
            return 0.0

        text_lower = candidate_text.lower()
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]

        word_matches = 0
        for w in query_words:
            if w in text_lower:
                word_matches += 1
        word_overlap = (word_matches / len(query_words)) if query_words else 0.0

        # Exact financial token matches (currencies, numbers, quarters, fiscal years)
        token_matches = 0
        if financial_tokens:
            for tok in financial_tokens:
                # Case-insensitive boundary match for tokens like AED, 50000, Q1
                if tok.lower() in text_lower:
                    token_matches += 1
            token_overlap = token_matches / len(financial_tokens)
        else:
            token_overlap = word_overlap

        # Weighted combination: 60% financial tokens / numbers, 40% query terms
        return round(0.6 * token_overlap + 0.4 * word_overlap, 4)

    @classmethod
    def rerank_and_filter(
        cls,
        request_id: str,
        query: str,
        candidates: list[RetrievedCandidate],
        financial_tokens: Optional[list[str]] = None,
        min_score_threshold: Optional[float] = None,
        max_candidates: int = 10,
    ) -> RerankingResult:
        """
        Execute Step 7 Reranking and Filtering:
        - Computes combined relevance score from vector score + entity match.
        - Deduplicates chunks by chunk_id and content signature.
        - Applies minimum relevance threshold.
        - Ranks and limits to max_candidates.
        """
        threshold = min_score_threshold if min_score_threshold is not None else cls.DEFAULT_MIN_THRESHOLD
        tokens = financial_tokens or []

        if not candidates:
            logger.info(f"[{request_id}] Zero candidates provided for reranking.")
            return RerankingResult(
                request_id=request_id,
                reranking_status="COMPLETED",
                results=[],
                result_count=0,
                filtered_out_count=0,
            )

        # 1. Deduplicate by chunk_id
        seen_chunks: set[str] = set()
        seen_texts: set[str] = set()
        unique_candidates: list[RetrievedCandidate] = []

        for cand in candidates:
            if cand.chunk_id in seen_chunks:
                continue
            seen_chunks.add(cand.chunk_id)

            # Secondary text deduplication
            normalized_snippet = cand.text_english[:120].strip() or cand.original_text[:120].strip()
            if normalized_snippet and normalized_snippet in seen_texts:
                continue
            if normalized_snippet:
                seen_texts.add(normalized_snippet)

            unique_candidates.append(cand)

        # 2. Score candidates
        scored_candidates: list[RetrievedCandidate] = []
        filtered_out = 0

        for cand in unique_candidates:
            combined_text = f"{cand.text_english}\n{cand.text_arabic}\n{cand.original_text}"
            overlap_score = cls._calculate_token_overlap_score(
                candidate_text=combined_text,
                query=query,
                financial_tokens=tokens,
            )

            # Combined formula: 70% vector similarity + 30% financial token/keyword overlap
            # Preserves original retrieval_score intact
            rel_score = round(cand.retrieval_score * 0.70 + overlap_score * 0.30, 4)
            cand.relevance_score = min(1.0, max(0.0, rel_score))

            if cand.relevance_score >= threshold:
                scored_candidates.append(cand)
            else:
                filtered_out += 1

        # 3. Sort by relevance_score descending
        scored_candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        # 4. Limit to max_candidates & assign rank
        final_candidates = scored_candidates[:max_candidates]
        for idx, cand in enumerate(final_candidates, start=1):
            cand.relevance_rank = idx

        logger.info(
            f"[{request_id}] Reranked {len(candidates)} candidates -> {len(final_candidates)} retained "
            f"(filtered out {filtered_out} below threshold {threshold})."
        )

        return RerankingResult(
            request_id=request_id,
            reranking_status="COMPLETED",
            results=final_candidates,
            result_count=len(final_candidates),
            filtered_out_count=filtered_out,
        )
