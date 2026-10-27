import csv
from pathlib import Path
from backend.ingestion.normalizers.schema import ExtractedItem


class CsvParser:
    """
    Step 3 Parser: Extracts headers, rows, and row references from CSV files.
    Preserves exact numeric values, signs, and currencies without flattening.
    """

    @classmethod
    def extract(cls, file_path: Path) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel

            reader = csv.reader(f, dialect=dialect)
            all_rows = [row for row in reader if any(cell.strip() for cell in row)]

        if not all_rows:
            return items

        if len(all_rows) > 1:
            headers = [cell.strip() for cell in all_rows[0]]
            data_rows = [[cell.strip() for cell in row] for row in all_rows[1:]]
            items.append(
                ExtractedItem(
                    type="table",
                    headers=headers,
                    rows=data_rows,
                    source={"row_start": 2, "row_end": len(all_rows), "total_rows": len(data_rows)},
                )
            )
        else:
            items.append(
                ExtractedItem(
                    type="text",
                    text=" | ".join(all_rows[0]),
                    source={"row": 1},
                )
            )

        return items
