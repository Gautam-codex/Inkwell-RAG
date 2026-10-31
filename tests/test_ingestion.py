"""Phase 2 tests: PDF parsing, table extraction and chunk metadata."""

import pymupdf
import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.pdf_parser import PDFParseError, parse_pdf
from src.ingestion.table_extractor import ExtractedTable, summarize_table, table_to_markdown

REQUIRED_KEYS = {"doc_id", "file_name", "page", "source_type", "chunk_id"}


# ---------- Parser ----------


def test_pages_are_numbered_from_one(sample_pdf):
    doc = parse_pdf(sample_pdf)
    assert [p.page for p in doc.pages] == [1, 2, 3, 4]


def test_heading_is_marked(sample_pdf):
    page1 = parse_pdf(sample_pdf).pages[0]
    assert page1.text.startswith("## Cooling System Overview")


def test_table_found_and_removed_from_text(sample_pdf):
    page2 = parse_pdf(sample_pdf).pages[1]
    assert len(page2.tables) == 1
    assert "| Quarter | Revenue | Profit |" in page2.tables[0].markdown
    assert "Revenue" not in page2.text  # table text is not duplicated in the page text


def test_small_images_skipped_large_kept(sample_pdf):
    doc = parse_pdf(sample_pdf)
    assert len(doc.pages[1].images) == 1  # 200x200 figure
    assert doc.pages[3].images == []  # 32x32 icon


def test_scanned_page_is_ocrd(sample_pdf):
    pytesseract = pytest.importorskip("pytesseract")
    from src.config import get_settings

    if get_settings().tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = get_settings().tesseract_cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip("Tesseract not installed")

    page3 = parse_pdf(sample_pdf).pages[2]
    assert page3.is_ocr
    assert "filter" in page3.text.lower()
    assert page3.images == []  # the scan itself is not treated as a figure


def test_corrupt_file_gives_readable_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(PDFParseError, match="not a valid PDF"):
        parse_pdf(bad)


def test_password_protected_gives_readable_error(tmp_path):
    path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    with pytest.raises(PDFParseError, match="password"):
        parse_pdf(path)


# ---------- Tables ----------


def test_table_to_markdown_cleans_cells():
    md, header = table_to_markdown([["Name", None], ["a|b", "line\nbreak"], [None, None]])
    assert header == ["Name", ""]
    assert md.splitlines() == ["| Name |  |", "| --- | --- |", "| a\\|b | line break |"]


def test_summary_falls_back_when_llm_fails():
    table = ExtractedTable(page=3, markdown="| A |\n| --- |\n| 1 |", n_rows=1, n_cols=1, bbox=(0, 0, 1, 1), header=["A"])

    def broken_llm(prompt: str) -> str:
        raise ConnectionError("Ollama not running")

    assert summarize_table(table, broken_llm) == "Table on page 3 with 1 rows and columns: A."
    assert summarize_table(table, lambda p: "Shows A.\nExtra line") == "Shows A."


# ---------- Chunker ----------


def test_chunk_metadata(sample_pdf):
    chunks = chunk_document(parse_pdf(sample_pdf), doc_id="abc123", chunk_size=300, chunk_overlap=50)
    assert chunks
    for c in chunks:
        assert REQUIRED_KEYS <= c.metadata.keys()
        assert c.metadata["doc_id"] == "abc123"
        assert c.metadata["file_name"] == "sample.pdf"
        assert None not in c.metadata.values()  # Chroma rejects None
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))  # chunk IDs are unique


def test_text_is_split_and_tables_are_not(sample_pdf):
    chunks = chunk_document(parse_pdf(sample_pdf), doc_id="abc123", chunk_size=300, chunk_overlap=50)
    text_chunks = [c for c in chunks if c.metadata["source_type"] == "text" and c.metadata["page"] == 1]
    assert len(text_chunks) > 1
    assert all(len(c.page_content) <= 300 for c in text_chunks)
    # The heading stays with the text below it instead of becoming a chunk on its own.
    assert text_chunks[0].page_content.startswith("## Cooling System Overview\nThe cooling")

    tables = [c for c in chunks if c.metadata["source_type"] == "table"]
    assert len(tables) == 1 and tables[0].metadata["page"] == 2
    assert tables[0].page_content.startswith("Table on page 2")  # summary first
    assert "| Q2 | 140 | 22 |" in tables[0].page_content  # whole table kept


def test_overlap_must_be_smaller_than_size(sample_pdf):
    with pytest.raises(ValueError):
        chunk_document(parse_pdf(sample_pdf), doc_id="x", chunk_size=100, chunk_overlap=100)
