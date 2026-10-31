"""Phase 6 tests: figure images are saved, described, chunked, and failures don't crash.

Fake vision models are used, so no Ollama is needed.
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import Settings
from src.ingestion.chunker import chunk_document
from src.ingestion.image_captioner import describe_figures
from src.ingestion.pdf_parser import parse_pdf

CHART_TEXT = "A bar chart titled Coolant flow by month. Flow peaks at 41 L/min in March."


def settings_for(tmp_path, **overrides) -> Settings:
    s = Settings(_env_file=None, data_dir=tmp_path, **overrides)
    s.ensure_dirs()
    return s


def test_figure_is_saved_described_and_chunked(tmp_path, sample_pdf):
    settings = settings_for(tmp_path)
    parsed = parse_pdf(sample_pdf)
    figures = describe_figures(parsed, "doc1", settings, vision_llm=FakeListChatModel(responses=[CHART_TEXT]))

    assert len(figures) == 1
    fig = figures[0]
    assert fig.page == 2
    assert fig.image_path == "extracted_images/doc1/p2_fig1.png"
    assert settings.data_file(fig.image_path).is_file()
    assert "Figure 1: Coolant flow by month" in fig.nearby_text  # the printed caption
    assert fig.description == CHART_TEXT

    chunks = chunk_document(parsed, "doc1", figures=figures)
    figure_chunks = [c for c in chunks if c.metadata["source_type"] == "figure"]
    assert len(figure_chunks) == 1
    meta = figure_chunks[0].metadata
    assert meta["page"] == 2 and meta["image_path"] == fig.image_path
    assert meta["chunk_id"] == "doc1:p2:figure:0"
    assert "Description: A bar chart" in figure_chunks[0].page_content


def test_decorative_images_are_dropped(tmp_path, sample_pdf):
    settings = settings_for(tmp_path)
    parsed = parse_pdf(sample_pdf)
    figures = describe_figures(parsed, "doc1", settings, vision_llm=FakeListChatModel(responses=["DECORATIVE"]))
    # The model said decorative, but the printed caption is still worth indexing.
    assert len(figures) == 1 and figures[0].description == ""


def test_vision_failure_does_not_crash(tmp_path, sample_pdf):
    class BrokenVision(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise ConnectionError("Ollama not running")

    settings = settings_for(tmp_path)
    figures = describe_figures(parse_pdf(sample_pdf), "doc1", settings, vision_llm=BrokenVision(responses=["x"]))
    assert len(figures) == 1  # kept thanks to its caption
    assert figures[0].description == "" and "Coolant flow" in figures[0].chunk_text()


def test_captioning_can_be_turned_off(tmp_path, sample_pdf):
    settings = settings_for(tmp_path, caption_figures=False)
    assert describe_figures(parse_pdf(sample_pdf), "doc1", settings) == []


def test_figure_limit(tmp_path, sample_pdf):
    settings = settings_for(tmp_path, max_figures_per_doc=0)
    assert describe_figures(parse_pdf(sample_pdf), "doc1", settings, vision_llm=FakeListChatModel(responses=["x"])) == []
