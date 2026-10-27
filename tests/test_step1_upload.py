import io
import uuid
import pytest
from httpx import AsyncClient
import fitz  # PyMuPDF


@pytest.mark.asyncio
async def test_upload_valid_pdf(async_client: AsyncClient):
    # Create a small valid PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Q1 Financial Report: Revenue was AED 500000.")
    pdf_bytes = doc.tobytes()
    doc.close()

    response = await async_client.post(
        "/api/v1/documents",
        files={"file": ("q1_report.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert "job_id" in data
    assert data["file_name"] == "q1_report.pdf"
    assert data["file_type"] == "pdf"
    assert data["status"] == "UPLOADED"
    assert data["job_status"] == "QUEUED"
    assert "file_uri" not in data  # Strictly verify no private file_uri leak


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_upload_unsupported_extension_rejected(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/documents",
        files={"file": ("script.exe", b"MZtest", "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_encrypted_pdf_rejected(async_client: AsyncClient):
    # Create an encrypted PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Secret Financial Data")
    pdf_bytes = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner123", user_pw="user123")
    doc.close()

    response = await async_client.post(
        "/api/v1/documents",
        files={"file": ("encrypted_report.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert "encrypted" in response.json()["detail"].lower()
