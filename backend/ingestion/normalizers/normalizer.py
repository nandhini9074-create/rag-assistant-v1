import uuid
from typing import Any
from backend.constants import ContentType
from backend.ingestion.normalizers.schema import ExtractedItem, NormalizedElement

NORMALIZATION_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d")


class DataNormalizer:
    """
    Step 4: Data Normalization.
    Transforms raw parser extracted items into typed NormalizedElement objects.
    Preserves financial values, decimal precision, currencies, and dates.
    Refines content types: 'table' -> 'financial_table', 'heading', 'narrative_text'.
    """

    @classmethod
    def _generate_element_id(cls, document_id: uuid.UUID, index: int, source: dict[str, Any]) -> uuid.UUID:
        seed = f"{document_id}:{index}:{sorted(source.items())}"
        return uuid.uuid5(NORMALIZATION_NAMESPACE, seed)

    @classmethod
    def format_markdown_table(cls, headers: list[str], rows: list[list[str]]) -> str:
        """
        Formats table into clean Markdown table text while strictly preserving cell values.
        """
        clean_headers = [h.replace("\n", " ").replace("|", "\\|").strip() for h in headers]
        header_line = " | ".join(clean_headers)

        row_lines = []
        for row in rows:
            clean_cells = [cell.replace("\n", " ").replace("|", "\\|").strip() for cell in row]
            if len(clean_cells) < len(clean_headers):
                clean_cells += [""] * (len(clean_headers) - len(clean_cells))
            row_lines.append(" | ".join(clean_cells[:len(clean_headers)]))

        return f"{header_line}\n" + "\n".join(row_lines)

    @classmethod
    def normalize(cls, document_id: uuid.UUID, extracted_items: list[ExtractedItem]) -> list[NormalizedElement]:
        elements: list[NormalizedElement] = []

        for idx, item in enumerate(extracted_items):
            element_id = cls._generate_element_id(document_id, idx, item.source)

            if item.type == "table" and item.headers and item.rows:
                content_type = ContentType.FINANCIAL_TABLE.value
                table_text = cls.format_markdown_table(item.headers, item.rows)
                elements.append(
                    NormalizedElement(
                        element_id=element_id,
                        content_type=content_type,
                        text=table_text,
                        source=item.source,
                        raw_data={"headers": item.headers, "rows": item.rows},
                    )
                )

            elif item.type == "heading":
                elements.append(
                    NormalizedElement(
                        element_id=element_id,
                        content_type=ContentType.HEADING.value,
                        text=item.text or "",
                        source=item.source,
                    )
                )

            else:
                text = (item.text or "").strip()
                if not text:
                    continue

                is_transaction = any(
                    k in text.lower() for k in ["txn", "transaction", "ref:", "receipt #", "inv:"]
                )
                content_type = ContentType.TRANSACTION_RECORD.value if is_transaction else ContentType.NARRATIVE_TEXT.value

                elements.append(
                    NormalizedElement(
                        element_id=element_id,
                        content_type=content_type,
                        text=text,
                        source=item.source,
                    )
                )

        return elements
