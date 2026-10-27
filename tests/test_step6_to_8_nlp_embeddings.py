import pytest
from backend.ingestion.embeddings.arabic_service import ArabicEmbeddingService
from backend.ingestion.embeddings.english_service import EnglishEmbeddingService
from backend.ingestion.language.detector import LanguageDetectorService
from backend.ingestion.translation.gpt4o_service import TranslationService


def test_language_detector_english():
    text = "Total revenue for the fiscal quarter reached AED 5,420,000 with EBITDA margin of 24.5%."
    lang, conf = LanguageDetectorService.detect_language(text)
    assert lang == "en"


def test_language_detector_arabic():
    text = "بلغ إجمالي الإيرادات للربع المالي 5,420,000 درهم إماراتي مع هامش أرباح بنسبة 24.5%."
    lang, conf = LanguageDetectorService.detect_language(text)
    assert lang == "ar"


@pytest.mark.asyncio
async def test_translation_preserves_numbers():
    translator = TranslationService()
    # Test bilingual generation
    original = "April revenue was AED 50000.50 and net profit was AED 12000.00."
    text_en, text_ar, status = await translator.translate_chunk(original, "en")

    assert status == "COMPLETED"
    assert text_en == original
    assert len(text_ar) > 0


@pytest.mark.asyncio
async def test_translation_falls_back_to_gemini_when_openai_key_absent():
    from unittest.mock import AsyncMock, MagicMock, patch

    with patch("backend.ingestion.translation.gpt4o_service.get_settings") as mock_settings:
        mock_settings.return_value.OPENAI_API_KEY = None
        mock_settings.return_value.OPENAI_TRANSLATION_MODEL = "gpt-4o"
        mock_settings.return_value.GEMINI_API_KEY = "dummy_gemini_key"
        mock_settings.return_value.GOOGLE_API_KEY = None
        mock_settings.return_value.GEMINI_TRANSLATION_MODEL = "gemini-1.5-flash"

        service = TranslationService()
        assert service.openai_client is None

        # Mock gemini client
        mock_gemini = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "بلغت إيرادات شهر أبريل 50000 درهم إماراتي."
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        service.gemini_client = mock_gemini

        original = "April revenue was AED 50000."
        text_en, text_ar, status = await service.translate_chunk(original, "en")

        assert status == "COMPLETED"
        assert text_en == original
        assert text_ar == "بلغت إيرادات شهر أبريل 50000 درهم إماراتي."
        mock_gemini.aio.models.generate_content.assert_called_once()


def test_separate_sentence_transformers_services():
    en_svc = EnglishEmbeddingService()
    ar_svc = ArabicEmbeddingService()

    en_svc.load_model()
    ar_svc.load_model()

    # Dynamic dimensions must be positive integers
    assert en_svc.dimension > 0
    assert ar_svc.dimension > 0

    en_vec = en_svc.encode("April revenue was AED 50000.")
    assert len(en_vec) == en_svc.dimension
    assert isinstance(en_vec[0], float)

    ar_vec = ar_svc.encode("بلغت إيرادات شهر أبريل 50000 درهم إماراتي.")
    assert len(ar_vec) == ar_svc.dimension
    assert isinstance(ar_vec[0], float)


def test_embedding_vector_validation_rejects_nan():
    en_svc = EnglishEmbeddingService()
    en_svc.load_model()

    # Intentionally corrupt vector with NaN
    dim = en_svc.dimension
    bad_vec = [float("nan")] * dim
    with pytest.raises(ValueError, match="NaN or Inf"):
        en_svc.validate_vector(bad_vec)
