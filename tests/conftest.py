"""Shared test fixtures. Builds a small PDF on the fly so tests need no sample files."""

import hashlib
import math
import re
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """A tiny stand-in for BGE so tests run in seconds with no download.

    Each word is hashed into one of 64 buckets and the bucket counts form the vector,
    so texts sharing words get similar vectors (enough to test the plumbing).
    """

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for word in re.findall(r"\w+", text.lower()):
            vector[hashlib.md5(word.encode()).digest()[0] % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

PARAGRAPH = (
    "The cooling system uses a closed water loop. Pump P1 circulates coolant at "
    "40 litres per minute through the heat exchanger. "
) * 12  # long enough to be split into several chunks


def _draw_table(page: pymupdf.Page, x: float, y: float, rows: list[list[str]]) -> None:
    """Draw a table with ruling lines, the kind pdfplumber detects by default."""
    col_w, row_h = 120, 22
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x + c * col_w, y + r * row_h, x + (c + 1) * col_w, y + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 7), cell, fontsize=10)


def _scanned_page_image(text: str) -> bytes:
    """Render text into a PNG, to fake a scanned page (no selectable text)."""
    tmp = pymupdf.open()
    page = tmp.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 400), text, fontsize=16)
    png = page.get_pixmap(dpi=150).tobytes("png")
    tmp.close()
    return png


def _bar_chart_png() -> bytes:
    """A small bar chart, so the vision model has a real figure to describe."""
    import io

    import matplotlib

    matplotlib.use("Agg")  # draw to memory, no window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    ax.bar(["Jan", "Feb", "Mar", "Apr"], [32, 38, 41, 36], color="#3060c0")
    ax.set_title("Coolant flow by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Flow (litres per minute)")
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def build_sample_pdf(path: Path) -> Path:
    """Create a 4-page PDF: text, table + chart, scanned page, page with only a tiny icon."""
    doc = pymupdf.open()

    # Page 1: heading + body text
    page = doc.new_page()
    page.insert_text((72, 80), "Cooling System Overview", fontsize=20)
    page.insert_textbox(pymupdf.Rect(72, 110, 540, 760), PARAGRAPH, fontsize=10)

    # Page 2: a ruled table and a bar chart (480x360 px) with a caption under it
    page = doc.new_page()
    page.insert_text((72, 60), "Quarterly results", fontsize=11)
    _draw_table(page, 72, 80, [["Quarter", "Revenue", "Profit"], ["Q1", "120", "15"], ["Q2", "140", "22"]])
    page.insert_image(pymupdf.Rect(72, 300, 312, 480), stream=_bar_chart_png())
    page.insert_text((72, 500), "Figure 1: Coolant flow by month", fontsize=10)

    # Page 3: scanned page = one big image, no text layer
    page = doc.new_page()
    page.insert_image(page.rect, stream=_scanned_page_image("Maintenance schedule: replace the filter every 6 months."))

    # Page 4: only a tiny 32x32 icon, which must be ignored
    page = doc.new_page()
    icon = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 32, 32), False)
    icon.set_rect(icon.irect, (200, 0, 0))
    page.insert_image(pymupdf.Rect(72, 72, 104, 104), pixmap=icon)

    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return build_sample_pdf(tmp_path_factory.mktemp("pdfs") / "sample.pdf")


if __name__ == "__main__":
    # python -m tests.conftest  -> writes sample.pdf in the current folder for manual testing
    print(build_sample_pdf(Path("sample.pdf")).resolve())
