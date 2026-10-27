import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryPreparationResult:
    """
    Structured result conforming to Step 4: Query Preparation / Preprocessing specification.
    """
    request_id: str
    original_query: str
    prepared_query: str
    query_language: str
    financial_tokens: list[str] = field(default_factory=list)
    status: str = "COMPLETED"


class QueryPreprocessor:
    """
    Step 4: Query Preparation / Preprocessing.
    Normalizes whitespace and clean formatting while strictly preserving:
    - Currency codes (AED, USD, SAR, EUR, GBP, EGP, etc.) and symbols ($, €, £, ¥)
    - Numbers, positive/negative signs, decimal points, percentages (%)
    - Dates, quarters (Q1, Q2, Q3, Q4), fiscal years (2025, 2026, FY24)
    - Arabic text, diacritics, and question markers (؟)
    - English text and case preservation for tickers/currencies
    """

    # Zero-width spaces, direction marks, and invisible formatting artifacts
    _INVISIBLE_CHARS_REGEX = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
    _WHITESPACE_REGEX = re.compile(r"\s+")

    # Regex matching financial entities, currency abbreviations, quarters, fiscal years, percentages, numbers
    _FINANCIAL_TOKENS_REGEX = re.compile(
        r"(?:\b(?:AED|USD|EUR|SAR|GBP|EGP|Q[1-4]|FY\d{2,4}|20\d{2})\b|[+\-]?(?:\$|€|£)?\d+(?:,\d+)*(?:\.\d+)?%?)",
        re.IGNORECASE,
    )

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Normalize excessive whitespace and invisible Unicode control characters
        while strictly avoiding destructive lowercasing or term loss.
        """
        if not text:
            return ""

        # Remove zero-width & directional control marks
        cleaned = cls._INVISIBLE_CHARS_REGEX.sub("", text)

        # Normalize multiple spaces, tabs, and newlines to a single space
        cleaned = cls._WHITESPACE_REGEX.sub(" ", cleaned).strip()

        return cleaned

    @classmethod
    def extract_financial_tokens(cls, query: str) -> list[str]:
        """
        Extract financial terms, numbers, currencies, dates for reranking and calculation validation.
        """
        if not query:
            return []
        tokens = cls._FINANCIAL_TOKENS_REGEX.findall(query)
        return [t.strip() for t in tokens if t.strip()]

    @classmethod
    def prepare_query(
        cls,
        request_id: str,
        query: str,
        query_language: str,
    ) -> QueryPreparationResult:
        """
        Execute Step 4 query preparation:
        Preserves original query meaning, avoids translation, normalizes layout,
        and extracts key financial tokens.
        """
        prepared = cls.clean_text(query)
        financial_tokens = cls.extract_financial_tokens(prepared)

        return QueryPreparationResult(
            request_id=request_id,
            original_query=query,
            prepared_query=prepared,
            query_language=query_language,
            financial_tokens=financial_tokens,
            status="COMPLETED",
        )
