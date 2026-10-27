import re


class TokenEstimator:
    """
    Fast and robust token counter supporting both English and Arabic text.
    Uses whitespace and unicode word boundary heuristics calibrated against BERT/RoBERTa tokenizers:
    - English: ~1 token per 3.5 - 4 characters or ~1.3 tokens per word.
    - Arabic: ~1.2 - 1.5 tokens per word due to clitics and affixes.
    """

    @classmethod
    def count_tokens(cls, text: str) -> int:
        if not text:
            return 0

        has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
        words = text.split()
        if not words:
            return 0

        if has_arabic:
            return max(1, int(len(words) * 1.4) + len(text) // 20)
        else:
            char_count = len(text)
            word_count = len(words)
            return max(1, int(word_count * 1.3), char_count // 4)
