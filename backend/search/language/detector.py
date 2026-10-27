import re
from typing import Optional
from pydantic import BaseModel, Field


class LanguageDetectionResult(BaseModel):
    """
    Step 3 Output Data Contract.
    """
    request_id: str
    original_query: str
    query_language: str = Field(..., description="'ar' or 'en'")
    response_language: str = Field(..., description="'ar' or 'en'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = "COMPLETED"


class QueryLanguageDetector:
    """
    Step 3: Query Language Detection.
    Detects whether the question is Arabic ('ar') or English ('en').
    Establishes the target response language.
    Applies defined fallback policies for ambiguous, short, or mixed-language queries.
    Preserves original query without alteration.
    """

    _detector = None

    @classmethod
    def _get_detector(cls):
        if cls._detector is None:
            try:
                from lingua import Language, LanguageDetectorBuilder
                cls._detector = LanguageDetectorBuilder.from_languages(
                    Language.ENGLISH, Language.ARABIC
                ).build()
            except Exception:
                cls._detector = "fallback"
        return cls._detector

    @classmethod
    def detect_language(cls, request_id: str, query: str) -> LanguageDetectionResult:
        """
        Analyzes the original query and determines query_language and response_language.
        Preserves original query unchanged.
        """
        cleaned = query.strip()
        if not cleaned:
            return LanguageDetectionResult(
                request_id=request_id,
                original_query=query,
                query_language="en",
                response_language="en",
                confidence=1.0,
                status="COMPLETED",
            )

        # Count character scripts
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", cleaned))
        latin_chars = len(re.findall(r"[a-zA-Z]", cleaned))

        # 1. Pure Arabic script
        if arabic_chars > 0 and latin_chars == 0:
            return LanguageDetectionResult(
                request_id=request_id,
                original_query=query,
                query_language="ar",
                response_language="ar",
                confidence=1.0,
                status="COMPLETED",
            )

        # 2. Pure Latin script
        if latin_chars > 0 and arabic_chars == 0:
            return LanguageDetectionResult(
                request_id=request_id,
                original_query=query,
                query_language="en",
                response_language="en",
                confidence=1.0,
                status="COMPLETED",
            )

        # 3. Use Lingua statistical detector for mixed / subtle queries
        detector = cls._get_detector()
        if detector != "fallback" and detector is not None:
            try:
                from lingua import Language
                confidences = detector.compute_language_confidence_values(cleaned)
                for conf in confidences:
                    if conf.language == Language.ARABIC and conf.value >= 0.5:
                        return LanguageDetectionResult(
                            request_id=request_id,
                            original_query=query,
                            query_language="ar",
                            response_language="ar",
                            confidence=round(conf.value, 3),
                            status="COMPLETED",
                        )
                    elif conf.language == Language.ENGLISH and conf.value >= 0.5:
                        return LanguageDetectionResult(
                            request_id=request_id,
                            original_query=query,
                            query_language="en",
                            response_language="en",
                            confidence=round(conf.value, 3),
                            status="COMPLETED",
                        )
            except Exception:
                pass

        # 4. Fallback policy for mixed code-switching queries:
        # Determine dominance based on count of alphabetic letters
        if arabic_chars > latin_chars:
            lang = "ar"
            conf = 0.85
        elif latin_chars > arabic_chars:
            lang = "en"
            conf = 0.85
        else:
            # Ambiguous (e.g. only numbers/symbols) -> default to en with low confidence
            lang = "en"
            conf = 0.50

        return LanguageDetectionResult(
            request_id=request_id,
            original_query=query,
            query_language=lang,
            response_language=lang,
            confidence=conf,
            status="COMPLETED",
        )
