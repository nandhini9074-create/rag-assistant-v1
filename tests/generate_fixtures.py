"""
Generates sample financial test fixtures in tests/fixtures/
"""
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_csv():
    csv_path = FIXTURES_DIR / "sample_revenue.csv"
    csv_content = """Quarter,Revenue,Expenses,Operating_Income,Currency
Q1-2026,500000.00,320000.00,180000.00,AED
Q2-2026,620000.50,380000.25,239999.25,AED
Q3-2026,710000.00,410000.00,300000.00,AED
Q4-2026,850000.75,490000.00,360000.75,AED
"""
    csv_path.write_text(csv_content, encoding="utf-8")
    print(f"Generated {csv_path}")


def generate_pdf():
    try:
        import fitz
        pdf_path = FIXTURES_DIR / "sample_financial_report.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Annual Financial Report 2026", fontsize=18)
        page.insert_text((50, 80), "The company generated revenue of AED 2680000.00 across all business lines.", fontsize=12)
        page.insert_text((50, 110), "Operating cash flow increased by 15.4% year-over-year.", fontsize=12)
        page.insert_text((50, 140), "Account ACC-98721 balance stands at USD 145000.00.", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        print(f"Generated {pdf_path}")
    except Exception as e:
        print(f"Could not generate PDF: {e}")


def generate_xlsx():
    try:
        import openpyxl
        xlsx_path = FIXTURES_DIR / "sample_balance_sheet.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Balance Sheet"
        ws.append(["Line Item", "2025 (AED)", "2026 (AED)", "Change (%)"])
        ws.append(["Cash and Cash Equivalents", "1200000", "1500000", "25.0"])
        ws.append(["Accounts Receivable", "850000", "920000", "8.2"])
        ws.append(["Total Current Assets", "2050000", "2420000", "18.0"])
        wb.save(str(xlsx_path))
        print(f"Generated {xlsx_path}")
    except Exception as e:
        print(f"Could not generate XLSX: {e}")


def generate_docx():
    try:
        import docx
        docx_path = FIXTURES_DIR / "sample_financial_policy.docx"
        doc = docx.Document()
        doc.add_heading("Financial Risk Policy", level=1)
        doc.add_paragraph("All cross-border transactions above USD 10000 require dual authorization.")
        table = doc.add_table(rows=1, cols=3)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Risk Level"
        hdr_cells[1].text = "Threshold (AED)"
        hdr_cells[2].text = "Approval Level"
        row_cells = table.add_row().cells
        row_cells[0].text = "Tier 1"
        row_cells[1].text = "500000.00"
        row_cells[2].text = "Chief Financial Officer"
        doc.save(str(docx_path))
        print(f"Generated {docx_path}")
    except Exception as e:
        print(f"Could not generate DOCX: {e}")


if __name__ == "__main__":
    generate_csv()
    generate_pdf()
    generate_xlsx()
    generate_docx()
