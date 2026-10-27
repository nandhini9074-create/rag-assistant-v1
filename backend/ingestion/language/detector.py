import re
from typing import Optional


class LanguageDetectorService:
    """
    Step 6: Language Detection.
    Detects chunk language independently using Lingua.
    Supports English ('en') and Arabic ('ar').
    Safe policy for numbers/tables/uncertain text: rejects silent wrong assignments.
    """

    _detector = None

    @classmethod
    def _get_detector(cls):
        if cls._detector is None:
            try:
                from lingua import Language, LanguageDetectorBuilder
                cls._detector = (
                    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.ARABIC)
                    .build()
                )
            except ImportError:
                cls._detector = "fallback"
        return cls._detector

    @classmethod
    def detect_language(cls, text: str) -> tuple[str, Optional[float]]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Cannot detect language of empty text")

        alpha_text = re.sub(r"[\d\W_]+", " ", cleaned_text).strip()

        if len(alpha_text) < 3:
            if re.search(r"[\u0600-\u06FF]", cleaned_text):
                return "ar", None
            return "en", None

        detector = cls._get_detector()
        if detector != "fallback":
            from lingua import Language
            detected_lang = detector.detect_language_of(alpha_text)
            confidence_values = detector.compute_language_confidence_values(alpha_text)

            top_conf: Optional[float] = None
            if confidence_values:
                top_conf = round(confidence_values[0].value, 4)

            if detected_lang == Language.ENGLISH:
                return "en", top_conf
            elif detected_lang == Language.ARABIC:
                return "ar", top_conf
            else:
                raise ValueError(f"Language detection returned unsupported or ambiguous language: {detected_lang}")
        else:
            has_arabic = bool(re.search(r"[\u0600-\u06FF]", alpha_text))
            if has_arabic:
                return "ar", None
            return "en", None
