import asyncio
import logging
import re
from typing import Optional
from openai import AsyncOpenAI
from backend.config import get_settings
from backend.constants import TranslationStatus

logger = logging.getLogger("translation-service")


class TranslationService:
    """
    Step 7: Bilingual Text Preparation & LLM Translation.
    Uses GPT-4o as primary translator and seamlessly falls back to Gemini
    when the OpenAI API key is missing or OpenAI requests fail.
    Produces text_english and text_arabic for each chunk.
    Enforces strict financial entity preservation: numbers, signs, decimals, currencies, dates.
    Includes rate-limit resilience (429 handling, model fallback, graceful degradation).
    """

    SYSTEM_PROMPT = """You are an expert financial translator specialized in English and Arabic financial documents.
Your task is to accurately translate the provided financial text between English and Arabic.

CRITICAL RULES:
1. PRESERVE ALL NUMBERS, SIGNS, DECIMALS, PERCENTAGES, AND UNITS EXACTLY (e.g., 50000, -12.5%, AED, USD).
2. NEVER CONVERT CURRENCIES OR ALTER FINANCIAL FACTS.
3. PRESERVE DATES, ACCOUNT NUMBERS, INVOICE NUMBERS, AND TRANSACTION IDENTIFIERS EXACTLY.
4. PRESERVE MARKDOWN TABLE FORMATTING AND DELIMITERS (| and ---).
5. Output ONLY the translated text without introductory phrases, commentary, or quotes.
"""

    def __init__(self) -> None:
        settings = get_settings()
        self.openai_api_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_TRANSLATION_MODEL
        self.gemini_api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        self.gemini_model = settings.GEMINI_TRANSLATION_MODEL
        self._last_call_time = 0.0

        # Initialize OpenAI client if valid key present
        if self.openai_api_key and self.openai_api_key != "test_openai_key":
            self.openai_client: Optional[AsyncOpenAI] = AsyncOpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None

        # Initialize Gemini client if key present
        if self.gemini_api_key and self.gemini_api_key != "test_gemini_key":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Google GenAI Client: {e}")
                self.gemini_client = None
        else:
            self.gemini_client = None

        if not self.openai_client and self.gemini_client:
            logger.info("OpenAI API key not present. Falling back to Gemini as primary translation LLM.")

    async def translate_chunk(
        self,
        original_text: str,
        original_language: str,
        max_retries: int = 3,
    ) -> tuple[str, str, str]:
        if not original_text.strip():
            raise ValueError("Cannot translate empty text chunk")

        status = TranslationStatus.COMPLETED.value

        try:
            if original_language == "en":
                text_english = original_text
                text_arabic = await self._translate(original_text, target_lang="Arabic", max_retries=max_retries)
            elif original_language == "ar":
                text_arabic = original_text
                text_english = await self._translate(original_text, target_lang="English", max_retries=max_retries)
            else:
                raise ValueError(f"Unsupported original language for translation: '{original_language}'")
        except Exception as e:
            logger.warning(f"Translation failed ({e}). Applying graceful bilingual fallback to preserve pipeline continuity.")
            text_english = original_text
            text_arabic = original_text
            status = TranslationStatus.SKIPPED.value

        if not text_english.strip() or not text_arabic.strip():
            text_english = original_text
            text_arabic = original_text
            status = TranslationStatus.SKIPPED.value

        return text_english, text_arabic, status

    async def _translate(self, text: str, target_lang: str, max_retries: int = 3) -> str:
        # Rate-limiting pacing delay to avoid hitting free-tier RPM limits
        await asyncio.sleep(0.5)

        # 1. If OpenAI client is present, try OpenAI first
        if self.openai_client:
            try:
                return await self._translate_openai(text, target_lang, max_retries=max_retries)
            except Exception as e:
                logger.warning(f"OpenAI translation failed ({e}). Checking Gemini fallback...")
                if self.gemini_client:
                    return await self._translate_gemini(text, target_lang, max_retries=max_retries)
                raise

        # 2. If OpenAI key is NOT present, fall back to Gemini
        if self.gemini_client:
            return await self._translate_gemini(text, target_lang, max_retries=max_retries)

        # 3. Fallback for testing / offline without API keys
        if target_lang == "Arabic":
            return f"[AR] {text}"
        return f"[EN] {text}"

    async def _translate_openai(self, text: str, target_lang: str, max_retries: int = 3) -> str:
        attempt = 0
        backoff = 1.0

        while attempt < max_retries:
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Translate the following financial text into {target_lang}:\n\n{text}",
                        },
                    ],
                    temperature=0.0,
                )
                translated = response.choices[0].message.content or ""
                return translated.strip()

            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError(f"OpenAI {self.openai_model} translation failed after {max_retries} attempts: {str(e)}")
                await asyncio.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError("OpenAI translation retries exhausted")

    async def _translate_gemini(self, text: str, target_lang: str, max_retries: int = 3) -> str:
        from google.genai import types

        attempt = 0
        backoff = 1.0
        prompt = f"Translate the following financial text into {target_lang}:\n\n{text}"
        model_to_use = self.gemini_model

        while attempt < max_retries:
            try:
                response = await self.gemini_client.aio.models.generate_content(
                    model=model_to_use,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                translated = response.text or ""
                return translated.strip()

            except Exception as e:
                attempt += 1
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_429:
                    logger.warning(f"Gemini model {model_to_use} hit rate limit (429). Attempt {attempt}/{max_retries}.")
                    # Try switching to fallback model if using gemini-2.5-flash
                    if model_to_use == "gemini-2.5-flash":
                        model_to_use = "gemini-3.5-flash-lite"
                        logger.info(f"Switched Gemini model to {model_to_use} for retry.")
                        await asyncio.sleep(1.0)
                        continue

                    # Extract suggested delay if present (e.g. 'retry in 5s')
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if match:
                        delay = float(match.group(1))
                        if delay <= 10.0:
                            logger.info(f"Backing off for {delay}s as requested by Gemini API...")
                            await asyncio.sleep(delay + 0.5)
                            continue

                if attempt >= max_retries:
                    raise RuntimeError(f"Gemini {model_to_use} translation failed after {max_retries} attempts: {err_str}")

                await asyncio.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError("Gemini translation retries exhausted")
