"""Figures and charts -> saved PNG + a factual description from a vision model.

For each large image found by the parser (logos/icons under MIN_IMAGE_SIZE are
already skipped):
  1. Crop that area of the page and save it as a PNG under
     data/extracted_images/<doc_id>/ (so the UI can show it next to the answer).
  2. Read the text printed just above and below it, which is usually the
     figure's own caption ("Figure 3: Revenue by quarter").
  3. Ask the vision model (llama3.2-vision / llava / gpt-4o-mini) to describe
     the chart: type, title, axes, labels, key values.
The description becomes a "figure" chunk, so a question like "what does the
revenue chart show?" can find it.

If no vision model is available, figures are not described (a warning is logged);
a figure that has a printed caption is still indexed using that caption alone.

Check the figures of a PDF from the command line:
    python -m src.ingestion.image_captioner sample.pdf
"""

import argparse
import base64
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pymupdf
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.config import Settings, get_settings
from src.ingestion.pdf_parser import ImageRef, ParsedDocument
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Saved images are scaled so the longest side is at most this many pixels.
# Bigger images are not described any better but are much slower on a CPU.
MAX_IMAGE_SIDE_PX = 1024
# How far above/below a figure to look for its caption, in points (72 points = 1 inch).
CAPTION_SEARCH_PT = 40
MAX_NEARBY_CHARS = 300

# The model answers with this word for logos, photos of people, decorations, etc.
DECORATIVE = "DECORATIVE"

FIGURE_PROMPT = """You are describing a figure from a document so that it can be found by search.
Describe only what is visible. Do not guess numbers you cannot read.

Include, when present:
- the type of figure (bar chart, line chart, pie chart, flow diagram, schematic, photo, ...)
- its title, axis labels and units, legend entries
- the key values, labels or steps shown, and the main trend or message

Write 3-6 factual sentences. If the image is only a logo, icon, signature or decoration,
reply with the single word {decorative}.

Text printed near the figure (may contain its caption): {nearby}"""


@dataclass
class FigureDescription:
    """One figure ready to become a chunk."""

    page: int
    image_path: str  # relative to the data folder, e.g. "extracted_images/<doc_id>/p4_fig1.png"
    description: str  # from the vision model ("" if none was available)
    nearby_text: str  # caption or text printed next to the figure

    def chunk_text(self) -> str:
        """The text that gets embedded and searched for this figure."""
        parts = [f"Figure on page {self.page}."]
        if self.nearby_text:
            parts.append(f"Caption / nearby text: {self.nearby_text}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Saving images and finding captions
# ---------------------------------------------------------------------------


def save_figure(page: pymupdf.Page, image: ImageRef, path: Path) -> None:
    """Crop the figure's area from the page and save it as PNG.

    Cropping the rendered page (instead of copying the raw image) keeps any labels
    or arrows that are drawn on top of the image, and handles every image format.
    """
    rect = pymupdf.Rect(image.bbox) & page.rect
    longest_pt = max(rect.width, rect.height, 1)
    dpi = max(72, min(200, int(MAX_IMAGE_SIDE_PX * 72 / longest_pt)))  # 72 pt per inch
    path.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(clip=rect, dpi=dpi).save(path)


def nearby_text(page: pymupdf.Page, image: ImageRef) -> str:
    """Text in a band just above and just below the figure (usually its caption)."""
    x0, y0, x1, y1 = image.bbox
    above = pymupdf.Rect(x0 - 20, y0 - CAPTION_SEARCH_PT, x1 + 20, y0)
    below = pymupdf.Rect(x0 - 20, y1, x1 + 20, y1 + CAPTION_SEARCH_PT)
    text = " ".join(page.get_textbox(r) for r in (above, below))
    return " ".join(text.split())[:MAX_NEARBY_CHARS]


# ---------------------------------------------------------------------------
# Vision model
# ---------------------------------------------------------------------------


def describe_image(image_path: Path, vision_llm: BaseChatModel, nearby: str = "") -> str:
    """Send one image to the vision model and return its description.

    Returns "" when the model says the image is decorative.
    """
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": FIGURE_PROMPT.format(decorative=DECORATIVE, nearby=nearby or "(none)")},
            # The same "data URL" format works for both Ollama and OpenAI.
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    )
    reply = str(vision_llm.invoke([message]).content).strip()
    if reply.upper().startswith(DECORATIVE):
        return ""
    return " ".join(reply.split())  # one paragraph, no stray line breaks


def _select_figures(parsed: ParsedDocument, limit: int) -> list[tuple[int, int, ImageRef]]:
    """(page, index on page, image) for each figure, skipping repeats of the same image."""
    selected, seen_xrefs = [], set()
    for page in parsed.pages:
        for i, image in enumerate(page.images):
            if image.xref and image.xref in seen_xrefs:
                continue  # e.g. the same banner on every page: describe it once
            seen_xrefs.add(image.xref)
            selected.append((page.page, i, image))
    if len(selected) > limit:
        logger.warning("%s has %d figures; only the first %d are described (MAX_FIGURES_PER_DOC)",
                       parsed.file_name, len(selected), limit)
    return selected[:limit]


def describe_figures(
    parsed: ParsedDocument,
    doc_id: str,
    settings: Optional[Settings] = None,
    vision_llm: Optional[BaseChatModel] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[FigureDescription]:
    """Save and describe every figure in a parsed PDF.

    Never raises because of the vision model: if it is missing or fails, the
    figure is kept only if it has a printed caption, and a warning is logged.
    """
    settings = settings or get_settings()
    if not settings.caption_figures:
        return []
    figures = _select_figures(parsed, settings.max_figures_per_doc)
    if not figures:
        return []

    if vision_llm is None:
        from src.generation.llm import check_vision, get_vision_llm

        problem = check_vision(settings)
        if problem:
            logger.warning("Figures will not be described by a vision model: %s", problem)
        else:
            vision_llm = get_vision_llm(settings)

    results: list[FigureDescription] = []
    doc = pymupdf.open(parsed.file_path)
    try:
        for done, (page_number, index, image) in enumerate(figures, start=1):
            page = doc[page_number - 1]
            relative = f"extracted_images/{doc_id}/p{page_number}_fig{index + 1}.png"
            path = settings.data_file(relative)
            save_figure(page, image, path)
            nearby = nearby_text(page, image)

            description = ""
            if vision_llm is not None:
                started = time.perf_counter()
                try:
                    description = describe_image(path, vision_llm, nearby)
                    logger.info("Described figure on page %d in %.1f s", page_number, time.perf_counter() - started)
                except Exception as exc:
                    logger.warning("Vision model failed on page %d, figure not described: %s", page_number, exc)

            if description or nearby:
                results.append(FigureDescription(page_number, relative, description, nearby))
            else:
                path.unlink(missing_ok=True)  # decorative, or nothing searchable about it
            if progress:
                progress(done, len(figures))
    finally:
        doc.close()

    logger.info("%s: %d of %d figures indexed", parsed.file_name, len(results), len(figures))
    return results


# ---------------------------------------------------------------------------
# Command-line check:  python -m src.ingestion.image_captioner sample.pdf
# ---------------------------------------------------------------------------


def main() -> None:
    from src.ingestion.pdf_parser import PDFParseError, parse_pdf
    from src.utils.hashing import doc_id_for_file

    parser = argparse.ArgumentParser(description="Extract and describe the figures in a PDF.")
    parser.add_argument("pdf", help="Path to a PDF file")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    try:
        parsed = parse_pdf(args.pdf)
    except PDFParseError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    settings = get_settings()
    started = time.perf_counter()
    figures = describe_figures(
        parsed, doc_id_for_file(args.pdf), settings,
        progress=lambda done, total: print(f"  described {done}/{total} figures", flush=True),
    )
    print(f"\n{parsed.file_name}: {len(figures)} figure chunk(s) in {time.perf_counter() - started:.0f} s\n")
    for fig in figures:
        print("=" * 70)
        print(f"Page {fig.page}  ->  {settings.data_file(fig.image_path)}")
        print("-" * 70)
        print(fig.chunk_text())
    print("=" * 70)


if __name__ == "__main__":
    main()
