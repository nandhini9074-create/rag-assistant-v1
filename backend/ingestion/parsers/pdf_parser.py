from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
from backend.ingestion.normalizers.schema import ExtractedItem


class PyMuPDFParser:
    """
    Step 3 Parser: Extracts text, headings, and tables from PDFs using PyMuPDF.
    Preserves exact financial values, signs, currencies, and page coordinates.
    """

    @classmethod
    def extract(cls, file_path: Path) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        doc = fitz.open(file_path)

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            table_rects: list[fitz.Rect] = []
            try:
                table_finder = page.find_tables()
                for table in table_finder:
                    df = table.extract()
                    if df and len(df) > 1:
                        raw_headers = [str(c or "").strip() for c in df[0]]
                        raw_rows = [[str(cell or "").strip() for cell in row] for row in df[1:]]

                        if any(raw_headers) and any(any(r) for r in raw_rows):
                            items.append(
                                ExtractedItem(
                                    type="table",
                                    headers=raw_headers,
                                    rows=raw_rows,
                                    source={"page": page_num, "bbox": list(table.bbox)},
                                )
                            )
                            table_rects.append(fitz.Rect(table.bbox))
            except Exception:
                pass

            blocks = page.get_text("blocks")
            for block in blocks:
                block_rect = fitz.Rect(block[:4])
                if any(block_rect.intersects(t_rect) for t_rect in table_rects):
                    continue

                text = block[4].strip()
                if not text:
                    continue

                is_heading = len(text.splitlines()) == 1 and len(text) < 120 and not text.endswith(".")
                items.append(
                    ExtractedItem(
                        type="heading" if is_heading else "text",
                        text=text,
                        source={
                            "page": page_num,
                            "bbox": [round(c, 2) for c in block[:4]],
                            "block_no": block[5],
                        },
                    )
                )

        doc.close()
        return items
