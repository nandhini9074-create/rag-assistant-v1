import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Sequence

logger = logging.getLogger("search-calculation")


@dataclass
class CalculationResult:
    """
    Deterministic Financial Calculation Result Contract.
    """
    performed: bool = False
    operation: Optional[str] = None  # "SUM", "DIFFERENCE", "AVERAGE"
    result_value: Optional[Decimal] = None
    currency: Optional[str] = None
    items: list[dict] = field(default_factory=list)
    formula_str: str = ""
    currency_mismatch: bool = False
    explanation: str = ""


class FinancialCalculationEngine:
    """
    Part 8: Deterministic Financial Calculation Engine.
    - Accurately detects calculation intent (total, sum, difference, average, margin).
    - Extracts monetary amounts with exact Decimal precision.
    - Prevents currency mixing (e.g. summing USD and AED without exchange rate).
    - Provides verified arithmetic proof to prevent LLM mathematical hallucinations.
    """

    _CALCULATION_INTENT_REGEX = re.compile(
        r"(?:\b(total|sum|aggregate|difference|margin|average|net|combined|all quarters|annual|yearly)\b|"
        r"إجمالي|مجموع|فرق|هامش|متوسط|صافي|الربع الأول والثاني|لكامل العام|سنوي)",
        re.IGNORECASE,
    )

    # Pattern extracting numbers associated with currencies (supporting commas, pipes, spaces)
    _MONEY_ITEM_REGEX = re.compile(
        r"(?:(AED|USD|EUR|SAR|GBP|EGP|\$|€|£)[,\s|]*([+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)|"
        r"([+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)[,\s|]+(AED|USD|EUR|SAR|GBP|EGP|درهم|دولار|ريال))",
        re.IGNORECASE,
    )

    @classmethod
    def has_calculation_intent(cls, query: str) -> bool:
        """
        Detects whether user is requesting arithmetic calculation, aggregation, or difference.
        """
        if not query:
            return False
        return bool(cls._CALCULATION_INTENT_REGEX.search(query))

    @classmethod
    def extract_monetary_items(cls, text: str) -> list[tuple[Decimal, str]]:
        """
        Extracts verified (Decimal amount, normalized currency) pairs from text.
        """
        items: list[tuple[Decimal, str]] = []
        if not text:
            return items

        for match in cls._MONEY_ITEM_REGEX.finditer(text):
            cur1, val1, val2, cur2 = match.groups()
            curr = cur1 or cur2 or ""
            val_str = val1 or val2 or ""

            # Normalize currency tags
            curr_upper = curr.upper().strip()
            if curr_upper in ("$", "دولار"):
                curr_norm = "USD"
            elif curr_upper in ("€", "يورو"):
                curr_norm = "EUR"
            elif curr_upper in ("درهم", "AED"):
                curr_norm = "AED"
            elif curr_upper in ("ريال", "SAR"):
                curr_norm = "SAR"
            else:
                curr_norm = curr_upper

            # Remove inner thousands commas
            clean_val = val_str.replace(",", "").strip()
            try:
                dec = Decimal(clean_val)
                items.append((dec, curr_norm))
            except (InvalidOperation, ValueError):
                continue

        return items

    @classmethod
    def compute(
        cls,
        request_id: str,
        query: str,
        context_chunks: Sequence[Any],
    ) -> CalculationResult:
        """
        Executes deterministic calculation when aggregation intent is detected in query.
        """
        if not cls.has_calculation_intent(query):
            return CalculationResult(performed=False)

        combined_text = "\n".join(getattr(c, "text_for_llm", str(c)) for c in context_chunks)
        monetary_items = cls.extract_monetary_items(combined_text)

        if len(monetary_items) < 2:
            return CalculationResult(
                performed=False,
                explanation="Less than two monetary values found in context; skipping aggregation.",
            )

        # Check for currency consistency
        currencies = {curr for _, curr in monetary_items if curr}
        if len(currencies) > 1:
            logger.warning(
                f"[{request_id}] Cannot perform deterministic aggregation across multiple currencies: {currencies}"
            )
            return CalculationResult(
                performed=False,
                currency_mismatch=True,
                explanation=f"Cannot compute aggregate across mismatched currencies ({', '.join(currencies)}) without an exchange rate.",
            )

        primary_currency = list(currencies)[0] if currencies else "UNSPECIFIED"
        values = [val for val, _ in monetary_items]

        query_lower = query.lower()
        if "average" in query_lower or "متوسط" in query:
            op = "AVERAGE"
            total = sum(values, Decimal("0"))
            res_val = round(total / Decimal(len(values)), 2)
            formula = f"({ ' + '.join(str(v) for v in values) }) / {len(values)} = {res_val} {primary_currency}"
        elif "difference" in query_lower or "فرق" in query:
            op = "DIFFERENCE"
            res_val = abs(values[0] - values[1])
            formula = f"|{values[0]} - {values[1]}| = {res_val} {primary_currency}"
        else:
            op = "SUM"
            res_val = sum(values, Decimal("0"))
            formula = f"{ ' + '.join(str(v) for v in values) } = {res_val} {primary_currency}"

        logger.info(f"[{request_id}] Deterministic calculation performed: {op} -> {formula}")

        return CalculationResult(
            performed=True,
            operation=op,
            result_value=res_val,
            currency=primary_currency,
            items=[{"value": str(v), "currency": c} for v, c in monetary_items],
            formula_str=formula,
            currency_mismatch=False,
            explanation=f"Deterministic {op} of {len(values)} line items: {formula}",
        )
