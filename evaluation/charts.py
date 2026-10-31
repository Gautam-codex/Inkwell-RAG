"""Bar charts for the evaluation report (matplotlib -> PNG).

Style rules used here:
  - one colour per retrieval set-up, always in the same order, so a set-up keeps
    its colour on every chart (colour-blind-safe order, checked with a validator)
  - value labels on bars + the same numbers in summary.md, because two of the
    colours are light and should not be the only way to read a value
  - one y-axis per chart; different units go in separate panels
  - light gridlines, no box around the plot
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402

# Fixed order: dense, BM25, hybrid, hybrid + re-ranker
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

# Mean vs p95 are two steps of one blue (not the set-up colours, to avoid confusion)
LATENCY = ["#2a78d6", "#86b6ef"]

MODE_ORDER = ["dense", "bm25", "hybrid", "hybrid_rerank"]
LABELS = {"dense": "Dense only", "bm25": "BM25 only", "hybrid": "Hybrid (RRF)", "hybrid_rerank": "Hybrid + re-ranker"}


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _ms_label(value: float) -> str:
    """Whole numbers, but keep one decimal for tiny values (BM25 takes well under 1 ms)."""
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def _grouped_bars(ax, groups: list[str], series: dict[str, list[float]], colors: list[str],
                  fmt: str = "{:.2f}") -> None:
    """Bars side by side within each group; a thin surface-coloured edge separates them."""
    n = len(series)
    width = 0.8 / n
    for i, ((name, values), color) in enumerate(zip(series.items(), colors)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width=width, color=color, label=name, edgecolor=SURFACE, linewidth=1.5)
        labels = [_ms_label(v) if fmt == "ms" else fmt.format(v) for v in values]
        ax.bar_label(bars, labels=labels, fontsize=7.5, color=INK_2, padding=2)
    ax.set_xticks(range(len(groups)), groups)


def _save(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def retrieval_comparison(path: Path, retrieval: list[dict]) -> None:
    by_mode = {r["mode"]: r for r in retrieval}
    modes = [m for m in MODE_ORDER if m in by_mode]
    metrics = [("hit_at_5", "Hit@5"), ("recall_at_5", "Recall@5"), ("mrr", "MRR")]
    series = {LABELS[m]: [by_mode[m][key] for key, _ in metrics] for m in modes}

    fig, ax = plt.subplots(figsize=(8, 4.2))
    _grouped_bars(ax, [label for _, label in metrics], series, SERIES[: len(modes)])
    _style(ax, "Retrieval quality by set-up (higher is better)", "Score (0-1)")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)


def retrieval_by_type(path: Path, by_type: list[dict]) -> None:
    types = [t for t in ("text", "table", "figure", "ocr") if any(r["type"] == t for r in by_type)]
    counts = {t: next(r["questions"] for r in by_type if r["type"] == t) for t in types}
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in by_type)]
    series = {}
    for m in modes:
        values = []
        for t in types:
            match = [r["hit_at_5"] for r in by_type if r["mode"] == m and r["type"] == t]
            values.append(match[0] if match else 0.0)
        series[LABELS[m]] = values

    fig, ax = plt.subplots(figsize=(8, 4.2))
    _grouped_bars(ax, [f"{t} (n={counts[t]})" for t in types], series, SERIES[: len(modes)])
    _style(ax, "Hit@5 by question type", "Hit@5 (0-1)")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)


def latency(path: Path, retrieval: list[dict], generation: Optional[dict]) -> None:
    """Retrieval latency per set-up (ms) and, if available, the full pipeline stages (s).

    Two panels with their own axes, because milliseconds and seconds would squash each other.
    """
    panels = 2 if generation else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.2), squeeze=False)
    by_mode = {r["mode"]: r for r in retrieval}
    modes = [m for m in MODE_ORDER if m in by_mode]

    ax = axes[0][0]
    _grouped_bars(ax, [LABELS[m].replace(" + ", "\n+ ") for m in modes],
                  {"Mean": [by_mode[m]["mean_latency_ms"] for m in modes],
                   "p95": [by_mode[m]["p95_latency_ms"] for m in modes]},
                  LATENCY, fmt="ms")
    _style(ax, "Retrieval latency per set-up", "Milliseconds")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=9)
    ax.margins(y=0.15)

    if generation:
        ax = axes[0][1]
        _grouped_bars(ax, ["Retrieval", "Generation"],
                      {"Mean": [generation["retrieval_mean_ms"] / 1000, generation["generation_mean_ms"] / 1000],
                       "p95": [generation["retrieval_p95_ms"] / 1000, generation["generation_p95_ms"] / 1000]},
                      LATENCY, fmt="{:.1f}")
        _style(ax, "Full pipeline per question", "Seconds")
        ax.legend(frameon=False, fontsize=9)
        ax.margins(y=0.15)
    _save(fig, path)


def answer_quality(path: Path, generation: dict) -> None:
    """Citation correctness by question type, and how the judge's 1-5 scores are spread."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax = axes[0]
    labels = list(generation["citation_by_type_pct"]) + ["all"]
    values = list(generation["citation_by_type_pct"].values()) + [generation["citation_correct_pct"]]
    bars = ax.bar(range(len(labels)), values, width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in values], fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(range(len(labels)), labels)
    _style(ax, "Citation correctness by question type", "Answers citing a correct page (%)")
    ax.set_ylim(0, 112)

    ax = axes[1]
    scores = [1, 2, 3, 4, 5]
    _grouped_bars(ax, [str(s) for s in scores],
                  {"Faithfulness": [generation["faithfulness_counts"][s] for s in scores],
                   "Correctness": [generation["correctness_counts"][s] for s in scores]},
                  SERIES[:2], fmt="{:.0f}")
    _style(ax, "LLM-judge scores (1 = worst, 5 = best)", "Number of answers")
    ax.set_xlabel("Score", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.margins(y=0.15)
    _save(fig, path)


def make_charts(reports_dir: Path, retrieval: list[dict], by_type: list[dict], generation: Optional[dict]) -> None:
    """Write all PNG charts into reports_dir."""
    retrieval_comparison(reports_dir / "retrieval_comparison.png", retrieval)
    retrieval_by_type(reports_dir / "retrieval_by_type.png", by_type)
    latency(reports_dir / "latency.png", retrieval, generation)
    if generation:
        answer_quality(reports_dir / "answer_quality.png", generation)
