from pathlib import Path
from openpyxl import load_workbook
from backend.ingestion.normalizers.schema import ExtractedItem


class XlsxParser:
    """
    Step 3 Parser: Extracts worksheets, tables, rows, and cell coordinates from XLSX.
    Preserves exact numeric values, signs, and currencies with data_only=True.
    """

    @classmethod
    def extract(cls, file_path: Path) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        wb = load_workbook(filename=file_path, data_only=True, read_only=True)

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[list[str]] = []

            for row in ws.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    formatted_row = [str(cell).strip() if cell is not None else "" for cell in row]
                    rows.append(formatted_row)

            if not rows:
                continue

            if len(rows) > 1:
                headers = rows[0]
                data_rows = rows[1:]
                items.append(
                    ExtractedItem(
                        type="table",
                        headers=headers,
                        rows=data_rows,
                        source={"sheet": sheet_name, "row_start": 2, "row_end": len(rows)},
                    )
                )
            elif len(rows) == 1:
                items.append(
                    ExtractedItem(
                        type="text",
                        text=" | ".join(rows[0]),
                        source={"sheet": sheet_name, "row": 1},
                    )
                )

        wb.close()
        return items
