"""Layout-aware PDF parsing, one page at a time.

For every page we collect:
  - text blocks (PyMuPDF), in reading order, with headings marked as "## ..."
  - tables (pdfplumber, see table_extractor.py); their text is removed from the
    normal page text so it is not indexed twice
  - images large enough to be figures (saved and captioned in Phase 6)
  - OCR text (pytesseract) when the page looks like a scan

Page numbers are 1-based everywhere, matching what a reader sees in a PDF viewer.

Check a PDF from the command line:
    python -m src.ingestion.pdf_parser sample.pdf
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pdfplumber
import pymupdf

from src.config import get_settings
from src.ingestion.table_extractor import BBox, ExtractedTable, extract_tables
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Called after each page as progress(pages_done, total_pages), so the UI can show a bar.
ProgressCallback = Callable[[int, int], None]

# A block counts as a heading if its font is this much bigger than the page's body text.
HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 150

# Render scanned pages at this resolution for OCR. 300 DPI is Tesseract's sweet spot.
OCR_DPI = 300


class PDFParseError(Exception):
    """Raised with a readable message when a PDF cannot be opened or read."""


@dataclass
class ImageRef:
    """An image on a page that is big enough to be a figure (not a logo or icon)."""

    xref: int  # PyMuPDF's internal ID for the image, used to extract it later
    width: int  # in pixels
    height: int
    bbox: BBox  # where it is drawn on the page, in points


@dataclass
class PageContent:
    """Everything extracted from one page."""

    page: int  # 1-based
    text: str
    is_ocr: bool = False
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """A whole parsed PDF."""

    file_path: Path
    file_name: str
    pages: list[PageContent]

    @property
    def page_count(self) -> int:
        return len(self.pages)


# ---------------------------------------------------------------------------
# Opening the file
# ---------------------------------------------------------------------------


def open_pdf(path: Path) -> pymupdf.Document:
    """Open a PDF with PyMuPDF, turning low-level errors into readable messages."""
    if not path.is_file():
        raise PDFParseError(f"File not found: {path}")
    try:
        # filetype="pdf" stops PyMuPDF from happily opening e.g. a .txt as a document.
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:
        logger.warning("Could not open %s: %s", path, exc)
        raise PDFParseError(f"'{path.name}' is not a valid PDF or is corrupted.") from exc

    if doc.needs_pass:
        doc.close()
        raise PDFParseError(f"'{path.name}' is password-protected. Remove the password and upload it again.")
    if doc.page_count == 0:
        doc.close()
        raise PDFParseError(f"'{path.name}' has no pages.")
    return doc


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _inside_any(bbox: BBox, regions: list[BBox], margin: float = 2.0) -> bool:
    """True if the centre of bbox lies inside one of the regions (e.g. a table)."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(
        r[0] - margin <= cx <= r[2] + margin and r[1] - margin <= cy <= r[3] + margin for r in regions
    )


def extract_text(page: pymupdf.Page, skip_regions: list[BBox]) -> str:
    """Return the page text block by block, in reading order, with headings marked.

    PyMuPDF's "dict" output gives blocks -> lines -> spans, and each span has a font
    size and style. We find the most common font size on the page (the body text)
    and treat noticeably bigger blocks as headings, writing them as "## Heading".

    sort=True orders blocks top-to-bottom, then left-to-right. This is right for
    single-column pages; on two-column pages lines from both columns can interleave.
    """
    data = page.get_text("dict", sort=True)
    blocks = []  # (text, max_font_size, all_bold, n_lines)
    size_counter: Counter = Counter()

    for block in data["blocks"]:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        if _inside_any(tuple(block["bbox"]), skip_regions):
            continue  # this text belongs to a table, which becomes its own chunk

        lines, sizes, all_bold = [], [], True
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if line_text:
                lines.append(line_text)
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(span["size"])
                    # Count characters per size so the body size wins over headings.
                    size_counter[round(span["size"], 1)] += len(span["text"])
                    all_bold = all_bold and bool(span["flags"] & 16)  # bit 16 = bold
        if lines:
            blocks.append(("\n".join(lines), max(sizes), all_bold, len(lines)))

    if not blocks:
        return ""

    body_size = size_counter.most_common(1)[0][0]
    output = ""
    for text, size, all_bold, n_lines in blocks:
        bigger = size >= body_size * HEADING_SIZE_RATIO
        bold_one_liner = all_bold and n_lines == 1 and len(text) < 80
        if (bigger or bold_one_liner) and len(text) <= HEADING_MAX_CHARS:
            # A heading ends with a single newline so it stays glued to the paragraph
            # below it; the splitter cuts at blank lines first and won't separate them.
            output += "## " + text.replace("\n", " ") + "\n"
        else:
            output += text + "\n\n"  # blank line between blocks = paragraph boundary
    return output.strip()


# ---------------------------------------------------------------------------
# Images and scanned pages
# ---------------------------------------------------------------------------


def _image_coverage(bbox: BBox, page_rect: pymupdf.Rect) -> float:
    """Fraction of the page area covered by an image (0.0 to 1.0)."""
    visible = pymupdf.Rect(bbox) & page_rect  # clip to the page
    if visible.is_empty or page_rect.get_area() == 0:
        return 0.0
    return visible.get_area() / page_rect.get_area()


def find_images(page: pymupdf.Page, min_size: int) -> tuple[list[ImageRef], float]:
    """Return (large images on the page, largest image coverage of the page).

    Images smaller than min_size pixels in width or height (logos, icons, bullets)
    are skipped. The coverage value is used to detect scanned pages.
    """
    images: list[ImageRef] = []
    seen_xrefs: set[int] = set()
    max_coverage = 0.0

    for info in page.get_image_info(xrefs=True):
        bbox = tuple(info["bbox"])
        max_coverage = max(max_coverage, _image_coverage(bbox, page.rect))

        xref = info.get("xref", 0)
        if info["width"] < min_size or info["height"] < min_size:
            continue
        if xref and xref in seen_xrefs:  # same image drawn twice on one page
            continue
        seen_xrefs.add(xref)
        images.append(ImageRef(xref=xref, width=info["width"], height=info["height"], bbox=bbox))
    return images, max_coverage


_tesseract_warned = False  # warn only once per run, not once per page


def ocr_page(page: pymupdf.Page) -> str:
    """Render the page to an image and read its text with Tesseract.

    Returns "" (and logs a warning) if Tesseract is not installed.
    """
    global _tesseract_warned
    import pytesseract
    from PIL import Image

    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    pix = page.get_pixmap(dpi=OCR_DPI)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError:
        if not _tesseract_warned:
            logger.warning("Tesseract is not installed, so scanned pages are skipped. See README.")
            _tesseract_warned = True
        return ""


# ---------------------------------------------------------------------------
# Whole document
# ---------------------------------------------------------------------------


def parse_page(page: pymupdf.Page, plumber_page, page_number: int) -> PageContent:
    """Extract text, tables, images and (if needed) OCR text from one page."""
    settings = get_settings()

    tables = extract_tables(plumber_page, page_number) if plumber_page is not None else []
    text = extract_text(page, skip_regions=[t.bbox for t in tables])
    images, max_coverage = find_images(page, settings.min_image_size)

    # A scanned page has (almost) no real text but one big image covering most of it.
    is_scan = len(text.strip()) < settings.ocr_min_text_chars and max_coverage >= settings.ocr_image_coverage
    is_ocr = False
    if is_scan:
        ocr_text = ocr_page(page)
        if ocr_text:
            text, is_ocr = ocr_text, True
            # The big image IS the scanned page, not a figure, so don't caption it.
            images = [im for im in images if _image_coverage(im.bbox, page.rect) < settings.ocr_image_coverage]
            logger.info("Page %d looks scanned, OCR read %d characters", page_number, len(text))

    return PageContent(page=page_number, text=text, is_ocr=is_ocr, tables=tables, images=images)


def parse_pdf(path: str | Path, progress: Optional[ProgressCallback] = None) -> ParsedDocument:
    """Parse a whole PDF page by page.

    Raises PDFParseError with a readable message for corrupt, empty or
    password-protected files. A single bad page is logged and skipped instead.
    """
    path = Path(path)
    doc = open_pdf(path)
    pages: list[PageContent] = []
    plumber = None

    try:
        # pdfplumber is only used for tables. If it can't open the file, we still
        # get the text and images from PyMuPDF.
        try:
            plumber = pdfplumber.open(path)
        except Exception as exc:
            logger.warning("pdfplumber could not open %s, tables skipped: %s", path.name, exc)

        total = doc.page_count
        for index in range(total):
            page_number = index + 1  # PyMuPDF counts from 0, people count from 1
            try:
                plumber_page = plumber.pages[index] if plumber is not None else None
                pages.append(parse_page(doc[index], plumber_page, page_number))
            except Exception as exc:
                logger.warning("Skipping page %d of %s: %s", page_number, path.name, exc)
                pages.append(PageContent(page=page_number, text=""))
            if progress:
                progress(page_number, total)
    finally:
        doc.close()
        if plumber is not None:
            plumber.close()

    logger.info(
        "Parsed %s: %d pages, %d tables, %d images, %d OCR pages",
        path.name,
        len(pages),
        sum(len(p.tables) for p in pages),
        sum(len(p.images) for p in pages),
        sum(p.is_ocr for p in pages),
    )
    return ParsedDocument(file_path=path, file_name=path.name, pages=pages)


# ---------------------------------------------------------------------------
# Command-line check:  python -m src.ingestion.pdf_parser sample.pdf
# ---------------------------------------------------------------------------


def main() -> None:
    """Print a per-page summary of a PDF so you can check the parser on your own files."""
    parser = argparse.ArgumentParser(description="Show what the parser extracts from each page of a PDF.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("--show-text", action="store_true", help="Also print the first 300 characters of each page")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")  # don't crash on odd characters in a Windows console

    try:
        doc = parse_pdf(args.pdf)
    except PDFParseError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # print() is fine here: this is the CLI's output, not logging.
    print(f"\n{doc.file_name}: {doc.page_count} pages\n")
    print(f"{'Page':>4} | {'Text chars':>10} | {'OCR':>3} | {'Tables':>6} | {'Images':>6}")
    print("-" * 44)
    for p in doc.pages:
        print(f"{p.page:>4} | {len(p.text):>10} | {'yes' if p.is_ocr else '':>3} | {len(p.tables):>6} | {len(p.images):>6}")
        if args.show_text and p.text:
            preview = p.text[:300].replace("\n", " ")
            print(f"       {preview}\n")
    print("-" * 44)
    print(
        f"{'All':>4} | {sum(len(p.text) for p in doc.pages):>10} | {sum(p.is_ocr for p in doc.pages):>3} | "
        f"{sum(len(p.tables) for p in doc.pages):>6} | {sum(len(p.images) for p in doc.pages):>6}"
    )


if __name__ == "__main__":
    main()
