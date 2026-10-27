from pathlib import Path
from typing import Any
from backend.config import get_settings
from backend.ingestion.normalizers.schema import ExtractedItem


class AzureDocumentIntelligenceParser:
    """
    Step 3 Parser: Extracts text and tables from scanned PDFs using Azure AI Document Intelligence.
    Uses 'prebuilt-layout' to capture headings, text, and tables.
    """

    @classmethod
    def extract(cls, file_path: Path) -> list[ExtractedItem]:
        settings = get_settings()
        endpoint = settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        key = settings.AZURE_DOCUMENT_INTELLIGENCE_KEY

        if not endpoint or not key:
            return cls._fallback_extract(file_path)

        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

            with open(file_path, "rb") as f:
                poller = client.begin_analyze_document(
                    "prebuilt-layout",
                    body=f,
                    content_type="application/octet-stream",
                )
            result = poller.result()

            items: list[ExtractedItem] = []

            if result.tables:
                for table in result.tables:
                    page_num = table.bounding_regions[0].page_number if table.bounding_regions else 1
                    col_count = table.column_count
                    row_count = table.row_count

                    grid: list[list[str]] = [["" for _ in range(col_count)] for _ in range(row_count)]
                    for cell in table.cells:
                        grid[cell.row_index][cell.column_index] = cell.content or ""

                    if row_count > 1:
                        headers = grid[0]
                        rows = grid[1:]
                        items.append(
                            ExtractedItem(
                                type="table",
                                headers=headers,
                                rows=rows,
                                source={"page": page_num, "table_id": getattr(table, "id", None)},
                            )
                        )

            if result.paragraphs:
                for p in result.paragraphs:
                    page_num = p.bounding_regions[0].page_number if p.bounding_regions else 1
                    role = getattr(p, "role", None)
                    item_type = "heading" if role in ("title", "sectionHeading") else "text"
                    items.append(
                        ExtractedItem(
                            type=item_type,
                            text=p.content,
                            source={"page": page_num, "role": role},
                        )
                    )

            return items

        except Exception:
            return cls._fallback_extract(file_path)

    @classmethod
    def _fallback_extract(cls, file_path: Path) -> list[ExtractedItem]:
        import fitz
        items: list[ExtractedItem] = []
        doc = fitz.open(file_path)
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if text:
                items.append(
                    ExtractedItem(
                        type="text",
                        text=text,
                        source={"page": page_num, "source": "local_fallback"},
                    )
                )
        doc.close()
        return items
