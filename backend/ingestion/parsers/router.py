import uuid
from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
from backend.constants import ParserType, PipelineStep, SupportedFileType


class FileRouter:
    """
    Step 2: File Routing.
    Inspects document type and content characteristics to determine the optimal parser.
    """

    @classmethod
    def inspect_pdf_text_density(cls, file_path: Path) -> tuple[bool, list[int], list[int]]:
        text_pages: list[int] = []
        scanned_pages: list[int] = []

        try:
            doc = fitz.open(file_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                if len(text) >= 50:
                    text_pages.append(page_idx + 1)
                else:
                    images = page.get_images()
                    if images:
                        scanned_pages.append(page_idx + 1)
                    elif len(text) > 0:
                        text_pages.append(page_idx + 1)
                    else:
                        scanned_pages.append(page_idx + 1)
            doc.close()
        except Exception:
            return False, [], []

        is_usable_text = len(text_pages) >= len(scanned_pages)
        return is_usable_text, text_pages, scanned_pages

    @classmethod
    def route_file(
        cls,
        document_id: uuid.UUID,
        job_id: uuid.UUID,
        file_type: str,
        file_path: Path,
    ) -> dict[str, Any]:
        normalized_type = file_type.lower().strip()

        if normalized_type == SupportedFileType.PDF.value:
            is_usable_text, text_pages, scanned_pages = cls.inspect_pdf_text_density(file_path)
            if is_usable_text:
                parser = ParserType.PYMUPDF
            else:
                parser = ParserType.AZURE_OCR
            routing_metadata = {
                "text_pages": text_pages,
                "scanned_pages": scanned_pages,
                "is_mixed": bool(text_pages and scanned_pages),
            }

        elif normalized_type == SupportedFileType.DOCX.value:
            parser = ParserType.PYTHON_DOCX
            routing_metadata = {}

        elif normalized_type == SupportedFileType.XLSX.value:
            parser = ParserType.OPENPYXL
            routing_metadata = {}

        elif normalized_type == SupportedFileType.CSV.value:
            parser = ParserType.CSV
            routing_metadata = {}

        else:
            raise ValueError(f"Unsupported file type for routing: '{file_type}'")

        return {
            "document_id": str(document_id),
            "job_id": str(job_id),
            "parser": parser.value,
            "status": "PROCESSING",
            "current_step": PipelineStep.FILE_ROUTING.value,
            "routing_metadata": routing_metadata,
        }
