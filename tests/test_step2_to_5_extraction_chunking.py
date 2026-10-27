import uuid
from pathlib import Path
import pytest
from backend.constants import ContentType, ParserType
from backend.ingestion.chunking.adaptive_chunker import AdaptiveChunker
from backend.ingestion.normalizers.normalizer import DataNormalizer
from backend.ingestion.normalizers.schema import ExtractedItem, NormalizedElement
from backend.ingestion.parsers.csv_parser import CsvParser
from backend.ingestion.parsers.router import FileRouter


def test_csv_parser_and_table_normalization(tmp_path: Path):
    # Create test CSV file
    csv_file = tmp_path / "financial_data.csv"
    csv_content = "Month,Revenue,Expenses,Net Profit,Currency\nApril,50000.00,32000.50,17999.50,AED\nMay,62000.00,40000.00,22000.00,AED\n"
    csv_file.write_text(csv_content, encoding="utf-8")

    # Step 2: Route file
    doc_id = uuid.uuid4()
    job_id = uuid.uuid4()
    routing = FileRouter.route_file(doc_id, job_id, "csv", csv_file)
    assert routing["parser"] == ParserType.CSV.value

    # Step 3: Extract items
    extracted = CsvParser.extract(csv_file)
    assert len(extracted) == 1
    table_item = extracted[0]
    assert table_item.type == "table"
    assert table_item.headers == ["Month", "Revenue", "Expenses", "Net Profit", "Currency"]
    assert len(table_item.rows) == 2
    assert table_item.rows[0] == ["April", "50000.00", "32000.50", "17999.50", "AED"]

    # Step 4: Normalize data
    normalized = DataNormalizer.normalize(doc_id, extracted)
    assert len(normalized) == 1
    norm_el = normalized[0]
    assert norm_el.content_type == ContentType.FINANCIAL_TABLE.value
    # Verify precision, currency, and headers preserved in markdown
    assert "Month | Revenue | Expenses | Net Profit | Currency" in norm_el.text
    assert "April | 50000.00 | 32000.50 | 17999.50 | AED" in norm_el.text


def test_adaptive_chunker_repeats_table_headers():
    doc_id = uuid.uuid4()
    headers = ["Account", "Description", "Debit", "Credit", "Date"]
    # 20 rows of financial transactions
    rows = [
        [f"ACC-{100+i}", f"Transaction item {i}", f"{1000 * i}.00", "0.00", f"2026-04-{i+1:02d}"]
        for i in range(20)
    ]

    table_text = DataNormalizer.format_markdown_table(headers, rows)
    elements = [
        NormalizedElement(
            element_id=uuid.uuid4(),
            content_type=ContentType.HEADING.value,
            text="General Ledger Q1",
            source={"page": 1},
        ),
        NormalizedElement(
            element_id=uuid.uuid4(),
            content_type=ContentType.FINANCIAL_TABLE.value,
            text=table_text,
            source={"page": 1},
            raw_data={"headers": headers, "rows": rows},
        ),
    ]

    # Use small max_chunk_tokens to force splitting into multiple chunks
    chunker = AdaptiveChunker(max_chunk_tokens=60, overlap_tokens=10)
    chunks = chunker.chunk(doc_id, elements)

    assert len(chunks) > 1

    # First chunk must contain heading and headers
    assert "General Ledger Q1" in chunks[0].text
    assert "Account | Description | Debit | Credit | Date" in chunks[0].text

    # Subsequent chunks MUST repeat the table headers
    for c in chunks[1:]:
        assert "Account | Description | Debit | Credit | Date" in c.text, "Chunk must repeat table headers!"

    # Verify deterministic IDs
    assert chunks[0].chunk_id == AdaptiveChunker.generate_chunk_id(doc_id, 0)
    assert chunks[0].qdrant_point_id == AdaptiveChunker.generate_point_id(doc_id, 0)
