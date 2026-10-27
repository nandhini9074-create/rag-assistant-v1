import io
import zipfile
from pathlib import Path
from typing import BinaryIO
import fitz  # PyMuPDF
from fastapi import HTTPException, status
from backend.constants import SupportedFileType


class FileValidator:
    """
    Validates uploaded files: extension, magic bytes, structure, corruption, and encryption.
    """

    ALLOWED_EXTENSIONS = {
        "pdf": SupportedFileType.PDF,
        "docx": SupportedFileType.DOCX,
        "xlsx": SupportedFileType.XLSX,
        "csv": SupportedFileType.CSV,
    }

    @classmethod
    def normalize_extension(cls, filename: str) -> SupportedFileType:
        if "." not in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename lacks a valid extension",
            )
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{ext}'. Supported formats: PDF, DOCX, XLSX, CSV",
            )
        return cls.ALLOWED_EXTENSIONS[ext]

    @classmethod
    def validate_content(cls, file_path: Path, file_type: SupportedFileType) -> None:
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty or missing",
            )

        with open(file_path, "rb") as f:
            header = f.read(16)
            f.seek(0)
            content = f.read()

        if file_type == SupportedFileType.PDF:
            if not header.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File header does not match PDF magic bytes",
                )
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                if doc.is_encrypted:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Encrypted or password-protected PDF files are not supported",
                    )
                if doc.page_count == 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="PDF file contains 0 pages",
                    )
                doc.close()
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupt or unreadable PDF file: {str(e)}",
                )

        elif file_type in (SupportedFileType.DOCX, SupportedFileType.XLSX):
            if not header.startswith(b"PK\x03\x04"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File content does not match {file_type.value.upper()} zip container signature",
                )
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    namelist = zf.namelist()
                    if file_type == SupportedFileType.DOCX and not any(n.startswith("word/") for n in namelist):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid DOCX container: missing 'word/' components",
                        )
                    if file_type == SupportedFileType.XLSX and not any(n.startswith("xl/") for n in namelist):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid XLSX container: missing 'xl/' components",
                        )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupt or invalid Office document: {str(e)}",
                )

        elif file_type == SupportedFileType.CSV:
            try:
                text_sample = content[:4096].decode("utf-8")
                if "\x00" in text_sample:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Binary content detected in CSV file",
                    )
            except UnicodeDecodeError:
                try:
                    content[:4096].decode("latin-1")
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Unable to decode CSV text content",
                    )
