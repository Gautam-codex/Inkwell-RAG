"""Table extraction with pdfplumber: tables -> Markdown + a one-line summary.

Each table becomes its own chunk and is never split, so the model always sees the
whole table. The summary is put in front of the Markdown so that semantic search
can find the table from a plain-English question ("what was revenue in 2023?").
"""

from dataclasses import dataclass
from typing import Callable, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# (x0, top, x1, bottom) in PDF points, origin at the top-left of the page.
BBox = tuple[float, float, float, float]

# A function that takes a prompt and returns the model's text. The real one is
# built from the LLM factory in Phase 5; until then a simple fallback is used.
Summarizer = Callable[[str], str]

TABLE_SUMMARY_PROMPT = (
    "Summarise the table below in ONE short sentence: what it shows and which "
    "columns or measures it contains. Do not add facts that are not in the table.\n\n{table}"
)


@dataclass
class ExtractedTable:
    """One table found on a page."""

    page: int  # 1-based page number
    markdown: str
    n_rows: int  # data rows, not counting the header
    n_cols: int
    bbox: BBox
    header: list[str]


def _clean_cell(cell: Optional[str]) -> str:
    """pdfplumber gives None for empty cells and keeps line breaks inside cells."""
    if cell is None:
        return ""
    # "|" would break the Markdown table, so escape it.
    return " ".join(str(cell).split()).replace("|", "\\|")


def table_to_markdown(rows: list[list[Optional[str]]]) -> tuple[str, list[str]]:
    """Convert raw rows to a Markdown table. The first row is used as the header.

    Returns (markdown, header). Returns ("", []) if the table has no real content.
    """
    cleaned = [[_clean_cell(c) for c in row] for row in rows if row]
    cleaned = [row for row in cleaned if any(row)]  # drop fully empty rows
    if len(cleaned) < 2:  # need a header and at least one data row
        return "", []

    n_cols = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (n_cols - len(row)) for row in cleaned]  # pad short rows

    header, body = cleaned[0], cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines), header


def extract_tables(plumber_page, page_number: int) -> list[ExtractedTable]:
    """Find tables on one pdfplumber page and convert each to Markdown.

    pdfplumber's default settings detect tables drawn with ruling lines. Tables
    laid out only with spacing (no lines) are often missed; their text is still
    indexed as normal page text, so nothing is lost.
    """
    tables: list[ExtractedTable] = []
    try:
        found = plumber_page.find_tables()
    except Exception as exc:  # malformed page content; keep going without tables
        logger.warning("Table detection failed on page %d: %s", page_number, exc)
        return tables

    for table in found:
        rows = table.extract()
        markdown, header = table_to_markdown(rows)
        if not markdown:
            continue
        tables.append(
            ExtractedTable(
                page=page_number,
                markdown=markdown,
                n_rows=markdown.count("\n") - 1,  # minus header and separator lines
                n_cols=len(header),
                bbox=tuple(table.bbox),
                header=header,
            )
        )
    return tables


def fallback_summary(table: ExtractedTable) -> str:
    """A summary built from the header alone, used when no LLM is available."""
    columns = ", ".join(h for h in table.header if h) or "unnamed columns"
    return f"Table on page {table.page} with {table.n_rows} rows and columns: {columns}."


def summarize_table(table: ExtractedTable, summarizer: Optional[Summarizer] = None) -> str:
    """Return a one-line summary of the table, using the LLM if one is given.

    If the LLM call fails (e.g. Ollama not running) we log a warning and fall back
    to the header-based summary instead of stopping the whole ingestion.
    """
    if summarizer is None:
        return fallback_summary(table)
    try:
        summary = summarizer(TABLE_SUMMARY_PROMPT.format(table=table.markdown)).strip()
        # Keep only the first line in case the model writes more than asked.
        summary = summary.splitlines()[0].strip() if summary else ""
        return summary or fallback_summary(table)
    except Exception as exc:
        logger.warning("Table summary failed on page %d, using fallback: %s", table.page, exc)
        return fallback_summary(table)
