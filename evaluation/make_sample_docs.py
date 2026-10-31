"""Build the 3 sample PDFs used by the evaluation set.

The documents are generated, so every fact (and the page it is on) is known
exactly. That makes the expected answers and pages in eval_set.json reliable.
Each PDF mixes the content types the system must handle:

  hydroflow_pump_manual.pdf     technical manual: text, 2 tables, a chart, a scanned page
  northwind_annual_report.pdf   financial report: text, 2 tables, a bar chart
  grid_storage_whitepaper.pdf   research whitepaper: text, a table, a line chart

Run:  python -m evaluation.make_sample_docs          (skips files that exist)
      python -m evaluation.make_sample_docs --force  (rebuild them)
"""

import argparse
import io
from pathlib import Path

import pymupdf

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_docs"

# Page geometry (A4 in points; 72 points = 1 inch)
LEFT, RIGHT, TOP, BOTTOM = 60, 535, 60, 790
WIDTH = RIGHT - LEFT

# Chart colours (validated colour-blind-safe categorical order) and neutral inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"


# ---------------------------------------------------------------------------
# A tiny layout helper: writes blocks top to bottom and tracks the position
# ---------------------------------------------------------------------------


class PageWriter:
    def __init__(self, doc: pymupdf.Document) -> None:
        self.page = doc.new_page()  # A4
        self.y = TOP

    def _text(self, text: str, size: float, font: str, gap: float) -> None:
        rect = pymupdf.Rect(LEFT, self.y, RIGHT, BOTTOM)
        unused = self.page.insert_textbox(rect, text, fontsize=size, fontname=font)
        if unused < 0:
            raise ValueError(f"Text does not fit on the page: {text[:40]}...")
        self.y += rect.height - unused + gap

    def title(self, text: str) -> None:
        self._text(text, 17, "hebo", 10)

    def heading(self, text: str) -> None:
        self._text(text, 13, "hebo", 4)

    def para(self, text: str) -> None:
        self._text(text, 10.5, "helv", 10)

    def table(self, rows: list[list[str]], col_widths: list[float] | None = None) -> None:
        """A table with ruling lines (pdfplumber detects these)."""
        n = len(rows[0])
        widths = col_widths or [WIDTH / n] * n
        row_h = 20
        for r, row in enumerate(rows):
            x = LEFT
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x, self.y, x + widths[c], self.y + row_h)
                self.page.draw_rect(rect, color=(0.3, 0.3, 0.3), width=0.6)
                font = "hebo" if r == 0 else "helv"
                self.page.insert_text((rect.x0 + 5, rect.y1 - 6), cell, fontsize=9.5, fontname=font)
                x += widths[c]
            self.y += row_h
        self.y += 14

    def image(self, png: bytes, width: float, height: float, caption: str) -> None:
        rect = pymupdf.Rect(LEFT, self.y, LEFT + width, self.y + height)
        self.page.insert_image(rect, stream=png)
        self.y += height + 6
        self._text(caption, 9.5, "heit", 14)


def scanned_page(doc: pymupdf.Document, text: str) -> None:
    """A page that is only a picture of text, like a scan (needs OCR)."""
    tmp = pymupdf.open()
    page = tmp.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 535, 780), text, fontsize=12, fontname="cour")
    png = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY).tobytes("png")
    tmp.close()
    target = doc.new_page()
    target.insert_image(target.rect, stream=png)


# ---------------------------------------------------------------------------
# Charts (matplotlib, rendered to PNG in memory)
# ---------------------------------------------------------------------------


def _figure(width_in: float = 6.0, height_in: float = 3.4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax, plt


def _png(fig, plt) -> bytes:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()


def pump_curve_png() -> bytes:
    fig, ax, plt = _figure()
    flow = [0, 5, 10, 15, 18, 20, 25]
    head = [45, 44, 41, 37, 34, 31, 21]
    ax.plot(flow, head, color=SERIES[0], linewidth=2, marker="o", markersize=5, label="HF-200 head")
    ax.annotate("Best efficiency point\n18 m³/h, 34 m", xy=(18, 34), xytext=(5, 25), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "->", "color": INK_2})
    ax.set_title("HF-200 performance curve at 2900 rpm", color=INK, fontsize=11)
    ax.set_xlabel("Flow rate (m³/h)", color=INK_2)
    ax.set_ylabel("Head (m)", color=INK_2)
    ax.set_ylim(0, 50)
    return _png(fig, plt)


def segment_revenue_png() -> bytes:
    fig, ax, plt = _figure()
    segments = ["Cloud", "Hardware", "Services"]
    y2023, y2024 = [180, 150, 100], [232, 146, 104]
    x = range(len(segments))
    bars1 = ax.bar([i - 0.2 for i in x], y2023, width=0.38, color=SERIES[0], label="2023", edgecolor=SURFACE, linewidth=2)
    bars2 = ax.bar([i + 0.2 for i in x], y2024, width=0.38, color=SERIES[1], label="2024", edgecolor=SURFACE, linewidth=2)
    for bars in (bars1, bars2):
        ax.bar_label(bars, fontsize=8, color=INK, padding=2)
    ax.set_xticks(list(x), segments)
    ax.set_title("Revenue by segment (EUR million)", color=INK, fontsize=11)
    ax.set_ylabel("EUR million", color=INK_2)
    ax.legend(frameon=False, fontsize=9)
    return _png(fig, plt)


def capacity_retention_png() -> bytes:
    fig, ax, plt = _figure()
    cycles = [0, 500, 1000, 1500, 2000, 2500, 3000]
    lines = {
        "LFP": [100, 99, 97.5, 96, 94.5, 93, 92],
        "Sodium-ion": [100, 98, 95.5, 93, 90.5, 88, 86],
        "NMC": [100, 97, 93.5, 90, 87, 84, 81],
    }
    for (name, values), color in zip(lines.items(), SERIES):
        ax.plot(cycles, values, color=color, linewidth=2, label=name)
        ax.annotate(f"{name} {values[-1]}%", xy=(cycles[-1], values[-1]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.axhline(80, color=INK_2, linewidth=1, linestyle="--")
    ax.text(50, 80.6, "End of life (80%)", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 3600)
    ax.set_ylim(75, 101)
    ax.set_title("Capacity retention vs cycle number (25 °C, 1C)", color=INK, fontsize=11)
    ax.set_xlabel("Cycle number", color=INK_2)
    ax.set_ylabel("Capacity retained (%)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    return _png(fig, plt)


# ---------------------------------------------------------------------------
# The three documents
# ---------------------------------------------------------------------------


def pump_manual(path: Path) -> None:
    doc = pymupdf.open()

    p = PageWriter(doc)  # page 1
    p.title("HydroFlow HF Series Centrifugal Pumps\nInstallation and Maintenance Manual")
    p.para(
        "The HydroFlow HF series are single-stage, end-suction centrifugal pumps designed for clean water, "
        "cooling circuits and light industrial liquids. This manual covers the HF-100, HF-200 and HF-300 "
        "models. Read it completely before installing or operating the pump and keep it near the installation."
    )
    p.para(
        "Every pump is hydrostatically tested at the factory at 1.5 times the maximum working pressure of "
        "16 bar, that is at 24 bar, before it is shipped."
    )
    p.heading("Intended use")
    p.para(
        "The pumps may be used with liquids at temperatures between -10 °C and 90 °C. They must not be used "
        "for flammable or explosive liquids, or for liquids containing abrasive solids larger than 2 mm."
    )
    p.heading("Warranty")
    p.para(
        "The warranty period is 24 months from the date of commissioning or 30 months from the date of "
        "delivery, whichever comes first. The warranty is void if non-original spare parts are used or if "
        "the pump is operated outside the limits given in this manual."
    )

    p = PageWriter(doc)  # page 2
    p.heading("Technical specifications")
    p.para("Table 1 lists the main ratings of each model at 50 Hz.")
    p.table([
        ["Model", "Max flow (m3/h)", "Max head (m)", "Motor power (kW)", "Weight (kg)"],
        ["HF-100", "12", "32", "2.2", "38"],
        ["HF-200", "25", "45", "5.5", "64"],
        ["HF-300", "40", "58", "11.0", "102"],
    ])
    p.para(
        "The noise level does not exceed 72 dB(A) at a distance of 1 m for any model. All motors are "
        "IE3 premium efficiency, protection class IP55."
    )

    p = PageWriter(doc)  # page 3
    p.heading("Installation")
    p.para(
        "Mount the pump on a flat concrete foundation whose mass is at least 1.5 times the mass of the pump. "
        "Leave a free space of at least 0.5 m on all sides of the pump for service access."
    )
    p.para(
        "The suction pipe must have a diameter at least equal to the pump inlet and should rise continuously "
        "towards the pump to avoid air pockets. Support all pipes independently so that no pipe forces act on "
        "the pump casing."
    )
    p.para(
        "After installation check the coupling alignment. The shaft misalignment must not exceed 0.05 mm; "
        "larger values cause bearing wear and seal leakage."
    )
    p.heading("Electrical connection")
    p.para(
        "The motor must be protected by a thermal overload relay set to the rated motor current. Connect the "
        "earth conductor before any other wire. Check the direction of rotation by briefly starting the motor."
    )

    p = PageWriter(doc)  # page 4
    p.heading("Performance curve")
    p.para("Figure 1 shows how the head of the HF-200 changes with the flow rate.")
    p.image(pump_curve_png(), 400, 227, "Figure 1: Performance curve of the HF-200 (head versus flow rate at 2900 rpm).")
    p.para(
        "The best efficiency point (BEP) of the HF-200 is at 18 m3/h and 34 m head. For a long service life, "
        "operate the pump between 70% and 110% of the BEP flow."
    )

    p = PageWriter(doc)  # page 5
    p.heading("Maintenance schedule")
    p.table([
        ["Task", "Interval"],
        ["Check mechanical seal for leaks", "Weekly"],
        ["Lubricate motor bearings", "Every 2,000 operating hours"],
        ["Inspect impeller and wear rings", "Every 8,000 operating hours"],
        ["Replace mechanical seal", "Every 16,000 hours or when leaking"],
    ], col_widths=[WIDTH * 0.55, WIDTH * 0.45])
    p.para(
        "Use lithium-complex grease of NLGI grade 2 for the motor bearings, 15 g per bearing. Always isolate "
        "the motor from the power supply before any maintenance work."
    )

    scanned_page(doc, (  # page 6: scanned troubleshooting sheet
        "TROUBLESHOOTING GUIDE\n\n"
        "Problem: Pump does not deliver water.\n"
        "Cause: Suction pipe and casing not primed.\n"
        "Remedy: Fill the pump casing and suction pipe with liquid.\n\n"
        "Problem: Excessive vibration.\n"
        "Cause: Impeller out of balance or cavitation.\n"
        "Remedy: Rebalance the impeller; increase the NPSH available.\n\n"
        "Problem: Motor overheats.\n"
        "Cause: Supply voltage deviates by more than 10 percent.\n"
        "Remedy: Check the supply voltage.\n\n"
        "Service hotline: +49 40 5550 1234"
    ))
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def annual_report(path: Path) -> None:
    doc = pymupdf.open()

    p = PageWriter(doc)  # page 1
    p.title("Northwind Systems\nAnnual Report 2024")
    p.heading("Letter from the Chief Executive Officer")
    p.para(
        "Dear shareholders, 2024 was a record year for Northwind Systems. Revenue grew by 12% to 482 million "
        "euros, driven above all by our cloud platform, which won more than 400 new enterprise customers."
    )
    p.para(
        "Our workforce grew to 3,150 employees in 14 countries. We also reduced the carbon emissions of our "
        "own operations by 18% compared with 2023."
    )
    p.para(
        "Given this strong result, the Board proposes a dividend of 1.20 euros per share for 2024, up from "
        "1.05 euros for 2023."
    )
    p.para("Maria Lindqvist, Chief Executive Officer")

    p = PageWriter(doc)  # page 2
    p.heading("Consolidated income statement")
    p.para("All amounts in EUR million unless stated otherwise.")
    p.table([
        ["Item", "2023", "2024"],
        ["Revenue", "430", "482"],
        ["Gross profit", "241", "275"],
        ["Operating income (EBIT)", "58", "71"],
        ["Net income", "41", "52"],
        ["Earnings per share (EUR)", "2.05", "2.60"],
    ], col_widths=[WIDTH * 0.5, WIDTH * 0.25, WIDTH * 0.25])
    p.para(
        "The gross margin improved from 56.0% to 57.1% because cloud subscriptions carry higher margins than "
        "hardware. The EBIT margin rose from 13.5% to 14.7%."
    )

    p = PageWriter(doc)  # page 3
    p.heading("Revenue by segment")
    p.image(segment_revenue_png(), 400, 227, "Figure 2: Revenue by business segment, 2023 and 2024 (EUR million).")
    p.para(
        "Cloud became our largest segment, with revenue of 232 million euros or 48% of the group total. "
        "Hardware revenue declined by 3% as customers moved to cloud services."
    )

    p = PageWriter(doc)  # page 4
    p.heading("Regional performance")
    p.table([
        ["Region", "Revenue 2024 (EUR million)", "Growth vs 2023"],
        ["Europe", "241", "9%"],
        ["North America", "145", "15%"],
        ["Asia-Pacific", "96", "17%"],
    ], col_widths=[WIDTH * 0.34, WIDTH * 0.40, WIDTH * 0.26])
    p.para(
        "Asia-Pacific was our fastest-growing region. We opened a new regional office in Singapore in "
        "March 2024 to serve customers in South-East Asia."
    )

    p = PageWriter(doc)  # page 5
    p.heading("Risks and outlook")
    p.para(
        "The main risks to our business are shortages of electronic components, movements in the US dollar "
        "exchange rate and cybersecurity threats against our cloud platform."
    )
    p.para(
        "For 2025 we expect revenue between 520 and 540 million euros and an EBIT margin of about 15%. "
        "Capital expenditure is planned at 35 million euros, mainly for two new data centres in Frankfurt "
        "and Dublin."
    )
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def whitepaper(path: Path) -> None:
    doc = pymupdf.open()

    p = PageWriter(doc)  # page 1
    p.title("Battery Chemistries for Grid-Scale Energy Storage:\nA Comparative Study")
    p.heading("Abstract")
    p.para(
        "This paper compares three lithium and sodium battery chemistries for stationary storage: lithium iron "
        "phosphate (LFP), nickel manganese cobalt (NMC) and sodium-ion. Cells were cycled for 3,000 cycles. "
        "LFP retained 92% of its capacity, sodium-ion 86% and NMC 81%."
    )
    p.heading("1. Introduction")
    p.para(
        "Grid operators need storage that lasts 15 to 20 years at a low cost per stored kilowatt-hour. Energy "
        "density matters less than in electric vehicles, while cycle life, cost and safety matter more."
    )

    p = PageWriter(doc)  # page 2
    p.heading("2. Methodology")
    p.para(
        "Twelve cells of each chemistry were cycled at 1C charge and discharge between 10% and 90% state of "
        "charge in a climate chamber held at 25 °C."
    )
    p.para(
        "Capacity was measured every 500 cycles with a slow C/3 reference test. End of life was defined as the "
        "point where a cell retains 80% of its initial capacity."
    )

    p = PageWriter(doc)  # page 3
    p.heading("3. Capacity retention")
    p.image(capacity_retention_png(), 400, 227,
            "Figure 3: Capacity retention versus cycle number for LFP, NMC and sodium-ion cells.")
    p.para(
        "NMC degraded fastest and is projected to reach the end-of-life threshold at about 3,200 cycles. LFP "
        "showed the slowest and most linear capacity fade."
    )

    p = PageWriter(doc)  # page 4
    p.heading("4. Cost and performance comparison")
    p.table([
        ["Property", "LFP", "NMC", "Sodium-ion"],
        ["Energy density (Wh/kg)", "165", "240", "140"],
        ["Cycle life to 80% (cycles)", "6,000", "3,200", "4,500"],
        ["Cell cost (USD/kWh)", "95", "120", "85"],
        ["Thermal runaway onset (C)", "270", "210", "300"],
    ], col_widths=[WIDTH * 0.40, WIDTH * 0.20, WIDTH * 0.20, WIDTH * 0.20])
    p.para(
        "NMC has the highest energy density but the shortest cycle life and the lowest thermal runaway "
        "temperature. Sodium-ion is the cheapest chemistry and the most resistant to thermal runaway."
    )

    p = PageWriter(doc)  # page 5
    p.heading("5. Conclusions")
    p.para(
        "LFP offers the best balance of cycle life, safety and cost for stationary storage today. We recommend "
        "LFP for new 4-hour storage systems and suggest re-evaluating sodium-ion in 2027, when larger "
        "production volumes are expected to improve its energy density."
    )
    doc.save(path, garbage=3, deflate=True)
    doc.close()


DOCUMENTS = {
    "hydroflow_pump_manual.pdf": pump_manual,
    "northwind_annual_report.pdf": annual_report,
    "grid_storage_whitepaper.pdf": whitepaper,
}


def build_all(force: bool = False) -> list[Path]:
    """Create the sample PDFs (only missing ones unless force=True)."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, builder in DOCUMENTS.items():
        path = SAMPLE_DIR / name
        if force or not path.exists():
            builder(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the evaluation sample PDFs.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the files exist")
    for built in build_all(parser.parse_args().force):
        print(built)
