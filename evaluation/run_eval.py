"""Evaluate the system on evaluation/eval_set.json and write a report.

What is measured
  1. Retrieval, for 4 set-ups (dense only, BM25 only, hybrid RRF, hybrid + re-ranker):
       Hit@5     share of questions with at least one correct page in the top 5
       Recall@5  share of the expected pages that appear in the top 5
       MRR       mean of 1 / (rank of the first correct result); 0 if none in top 5
     "Correct" = a retrieved chunk from the expected file AND one of the expected pages.
  2. Citation correctness: share of answers whose cited pages include an expected page.
  3. Answer quality: an LLM judge scores faithfulness (claims supported by the
     sources) and correctness (agrees with the expected answer), 1-5.
  4. Latency (mean and p95) for retrieval and generation, and peak RAM of this
     process and of Ollama.

Output (evaluation/reports/): summary.md, CSV files and PNG charts.

Run:
    python -m evaluation.run_eval                     full run (about 15-25 min on a CPU)
    python -m evaluation.run_eval --skip-generation   retrieval only (about 2 min)
    python -m evaluation.run_eval --limit 5           quick try on 5 questions
"""

import argparse
import csv
import json
import math
import platform
import re
import shutil
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

from src.config import Settings, get_settings
from src.utils.logging import get_logger, set_console_level

logger = get_logger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / "eval_set.json"
SAMPLE_DIR = EVAL_DIR / "sample_docs"
REPORTS_DIR = EVAL_DIR / "reports"
EVAL_DATA_DIR = EVAL_DIR / "eval_data"  # separate index, so the app's own data/ is untouched

MODE_LABELS = {
    "dense": "Dense only",
    "bm25": "BM25 only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + re-ranker",
}
QUESTION_TYPES = ("text", "table", "figure", "ocr")


@dataclass
class EvalQuestion:
    id: str
    type: str
    question: str
    expected_answer: str
    expected_file: str
    expected_pages: list[int]


def load_eval_set(path: Path = EVAL_SET) -> tuple[list[str], list[EvalQuestion]]:
    """Return (document file names, questions) from the JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["documents"], [EvalQuestion(**q) for q in data["questions"]]


# ---------------------------------------------------------------------------
# Metrics (plain functions, unit-tested in tests/test_evaluation.py)
# ---------------------------------------------------------------------------


def relevance_flags(retrieved: list[tuple[str, int]], expected_file: str, expected_pages: list[int]) -> list[bool]:
    """For each retrieved (file, page), in rank order: is it a correct page?"""
    return [f == expected_file and p in expected_pages for f, p in retrieved]


def hit_at_k(flags: list[bool], k: int) -> float:
    """1.0 if any of the first k results is correct, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0


def recall_at_k(retrieved: list[tuple[str, int]], expected_file: str, expected_pages: list[int], k: int) -> float:
    """Share of the expected pages found among the first k results."""
    found = {p for f, p in retrieved[:k] if f == expected_file and p in expected_pages}
    return len(found) / len(set(expected_pages))


def reciprocal_rank(flags: list[bool], k: int) -> float:
    """1 / rank of the first correct result within the first k (rank starts at 1), else 0."""
    for rank, ok in enumerate(flags[:k], start=1):
        if ok:
            return 1.0 / rank
    return 0.0


def p95(values: list[float]) -> float:
    """95th percentile, nearest-rank method: the value below which 95% of results fall."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def parse_judge_scores(text: str) -> tuple[Optional[int], Optional[int]]:
    """Read 'FAITHFULNESS: 4' and 'CORRECTNESS: 5' from the judge's reply."""
    def find(name: str) -> Optional[int]:
        match = re.search(rf"{name}\W*([1-5])", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    return find("FAITHFULNESS"), find("CORRECTNESS")


# ---------------------------------------------------------------------------
# Peak RAM, sampled in the background
# ---------------------------------------------------------------------------


class RamMonitor:
    """Samples memory once a second: this Python process and all Ollama processes.

    Ollama runs the LLM in separate processes, so its memory is measured on its own.
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.peak_app_mb = self.peak_ollama_mb = self.peak_total_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._me = psutil.Process()

    @staticmethod
    def _ollama_mb() -> float:
        total = 0
        for proc in psutil.process_iter(["name", "memory_info"]):
            name = (proc.info.get("name") or "").lower()
            if ("ollama" in name or "llama" in name) and proc.info.get("memory_info"):
                total += proc.info["memory_info"].rss
        return total / 1e6

    def sample(self) -> None:
        app = self._me.memory_info().rss / 1e6
        ollama = self._ollama_mb()
        self.peak_app_mb = max(self.peak_app_mb, app)
        self.peak_ollama_mb = max(self.peak_ollama_mb, ollama)
        self.peak_total_mb = max(self.peak_total_mb, app + ollama)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample()
            except Exception:  # a process ended while we were reading it
                pass

    def __enter__(self) -> "RamMonitor":
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def eval_settings(no_figures: bool) -> Settings:
    """The app's settings (.env), but with a separate data folder for the evaluation index."""
    update: dict = {"data_dir": EVAL_DATA_DIR}
    if no_figures:
        update["caption_figures"] = False
    settings = get_settings().model_copy(update=update)
    settings.ensure_dirs()
    return settings


def build_index(settings: Settings, documents: list[str], rebuild: bool):
    """Index the sample PDFs (already indexed files are skipped)."""
    from evaluation.make_sample_docs import build_all
    from src.indexing.indexer import Indexer

    if rebuild and EVAL_DATA_DIR.exists():
        shutil.rmtree(EVAL_DATA_DIR)
        settings.ensure_dirs()
    build_all()
    indexer = Indexer(settings)
    for name in documents:
        path = SAMPLE_DIR / name
        print(f"  indexing {name} ...", flush=True)
        result = indexer.index_file(path)
        print(f"    {result.status}: {result.counts or result.message}")
        if result.status == "error":
            sys.exit(f"Could not index {name}: {result.message}")
    return indexer


def evaluate_retrieval(retriever, questions: list[EvalQuestion], k: int) -> list[dict]:
    """Run every question through all 4 retrieval set-ups."""
    from src.retrieval.hybrid_retriever import MODES

    rows = []
    for mode in MODES:
        print(f"  {MODE_LABELS[mode]:<20}", end="", flush=True)
        for q in questions:
            result = retriever.retrieve(q.question, mode=mode, top_n=k)
            retrieved = [(c.metadata["file_name"], c.metadata["page"]) for c in result.chunks]
            flags = relevance_flags(retrieved, q.expected_file, q.expected_pages)
            rows.append({
                "mode": mode, "id": q.id, "type": q.type,
                "hit": hit_at_k(flags, k),
                "recall": recall_at_k(retrieved, q.expected_file, q.expected_pages, k),
                "rr": reciprocal_rank(flags, k),
                "latency_ms": round(result.timings_ms.get("total_ms", 0.0), 1),
                "retrieved": "; ".join(f"{f} p{p}" for f, p in retrieved),
            })
            print(".", end="", flush=True)
        print()
    return rows


def evaluate_generation(chain, judge_chain, questions: list[EvalQuestion]) -> list[dict]:
    """Answer every question with the full pipeline, check citations, ask the judge."""
    from src.generation.rag_chain import GenerationError, build_context
    from src.retrieval.hybrid_retriever import RetrievalError

    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {q.question}", flush=True)
        row = {"id": q.id, "type": q.type, "question": q.question, "expected_answer": q.expected_answer,
               "expected": f"{q.expected_file} p{q.expected_pages}"}
        try:
            answer = chain.ask(q.question)
        except (RetrievalError, GenerationError) as exc:
            rows.append({**row, "answer": f"ERROR: {exc}", "cited": "", "citation_correct": 0,
                         "faithfulness": None, "correctness": None, "retrieval_ms": None, "generation_ms": None})
            continue

        cited = [(c.file_name, c.page) for c in answer.citations]
        citation_ok = any(f == q.expected_file and p in q.expected_pages for f, p in cited)

        context, _ = build_context(answer.sources)
        try:
            verdict = judge_chain.invoke({"context": context, "question": q.question,
                                          "expected": q.expected_answer, "answer": answer.text})
        except Exception as exc:
            logger.warning("Judge failed on %s: %s", q.id, exc)
            verdict = ""
        faithfulness, correctness = parse_judge_scores(verdict)

        rows.append({
            **row,
            "answer": answer.text,
            "cited": "; ".join(f"{f} p{p}" for f, p in cited),
            "citation_correct": int(citation_ok),
            "faithfulness": faithfulness,
            "correctness": correctness,
            "retrieval_ms": round(answer.timings_ms.get("retrieval_ms", 0.0), 1),
            "generation_ms": round(answer.timings_ms.get("generation_ms", 0.0), 1),
        })
        print(f"      cited {rows[-1]['cited'] or '-'} | citation ok: {citation_ok} | "
              f"faithfulness {faithfulness} | correctness {correctness}")
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_retrieval(rows: list[dict]) -> list[dict]:
    summary = []
    for mode, label in MODE_LABELS.items():
        mine = [r for r in rows if r["mode"] == mode]
        if not mine:
            continue
        latencies = [r["latency_ms"] for r in mine]
        summary.append({
            "mode": mode, "setup": label,
            "hit_at_5": round(mean([r["hit"] for r in mine]), 3),
            "recall_at_5": round(mean([r["recall"] for r in mine]), 3),
            "mrr": round(mean([r["rr"] for r in mine]), 3),
            "mean_latency_ms": round(mean(latencies), 1),
            "p95_latency_ms": round(p95(latencies), 1),
        })
    return summary


def summarize_by_type(rows: list[dict]) -> list[dict]:
    summary = []
    for mode, label in MODE_LABELS.items():
        for qtype in QUESTION_TYPES:
            mine = [r for r in rows if r["mode"] == mode and r["type"] == qtype]
            if mine:
                summary.append({"mode": mode, "setup": label, "type": qtype, "questions": len(mine),
                                "hit_at_5": round(mean([r["hit"] for r in mine]), 3),
                                "mrr": round(mean([r["rr"] for r in mine]), 3)})
    return summary


def summarize_generation(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["retrieval_ms"] is not None]
    faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    corr = [r["correctness"] for r in rows if r["correctness"] is not None]
    retrieval = [r["retrieval_ms"] for r in ok]
    generation = [r["generation_ms"] for r in ok]
    by_type = {}
    for qtype in QUESTION_TYPES:
        mine = [r for r in rows if r["type"] == qtype]
        if mine:
            by_type[qtype] = round(100 * mean([r["citation_correct"] for r in mine]), 1)
    return {
        "answered": len(ok), "total": len(rows),
        "citation_correct_pct": round(100 * mean([r["citation_correct"] for r in rows]), 1),
        "citation_by_type_pct": by_type,
        "faithfulness_mean": round(mean(faith), 2), "faithfulness_scored": len(faith),
        "correctness_mean": round(mean(corr), 2), "correctness_scored": len(corr),
        "faithfulness_counts": {s: faith.count(s) for s in range(1, 6)},
        "correctness_counts": {s: corr.count(s) for s in range(1, 6)},
        "retrieval_mean_ms": round(mean(retrieval), 1), "retrieval_p95_ms": round(p95(retrieval), 1),
        "generation_mean_ms": round(mean(generation), 1), "generation_p95_ms": round(p95(generation), 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: dict[str, str]) -> str:
    """Markdown table from dict rows; columns maps key -> header."""
    lines = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in columns) + " |")
    return "\n".join(lines)


def write_summary_md(path: Path, info: dict, retrieval: list[dict], by_type: list[dict],
                     generation: Optional[dict], ram: RamMonitor) -> None:
    parts = [
        "# Evaluation report",
        "",
        f"Run on {info['date']} · {info['questions']} questions over {info['documents']} documents · "
        f"top-k = {info['k']}",
        "",
        f"Models: LLM `{info['llm']}` · embeddings `{info['embeddings']}` · re-ranker `{info['reranker']}` · "
        f"vision `{info['vision']}`",
        "",
        f"Machine: {info['machine']}",
        "",
        "## 1. Retrieval: comparing the four set-ups",
        "",
        "A result counts as correct when it comes from the expected file and one of the expected pages.",
        "",
        md_table(retrieval, {"setup": "Set-up", "hit_at_5": "Hit@5", "recall_at_5": "Recall@5", "mrr": "MRR",
                             "mean_latency_ms": "Mean latency (ms)", "p95_latency_ms": "p95 latency (ms)"}),
        "",
        "![Retrieval comparison](retrieval_comparison.png)",
        "",
        "### Hit@5 by question type",
        "",
    ]
    types_present = [t for t in QUESTION_TYPES if any(r["type"] == t for r in by_type)]
    pivot = []
    for mode, label in MODE_LABELS.items():
        row = {"setup": label}
        for t in types_present:
            match = [r for r in by_type if r["mode"] == mode and r["type"] == t]
            row[t] = match[0]["hit_at_5"] if match else "-"
        pivot.append(row)
    counts = {t: next(r["questions"] for r in by_type if r["type"] == t) for t in types_present}
    parts += [md_table(pivot, {"setup": "Set-up", **{t: f"{t} (n={counts[t]})" for t in types_present}}), "",
              "![Hit@5 by question type](retrieval_by_type.png)", ""]

    if generation:
        g = generation
        parts += [
            "## 2. Answers (full pipeline: hybrid + re-ranker + LLM)",
            "",
            md_table([
                {"m": "Questions answered without error", "v": f"{g['answered']} / {g['total']}"},
                {"m": "Citation correctness (cited page includes an expected page)", "v": f"{g['citation_correct_pct']}%"},
                {"m": "Faithfulness, LLM judge (1-5)", "v": f"{g['faithfulness_mean']} (n={g['faithfulness_scored']})"},
                {"m": "Correctness vs expected answer, LLM judge (1-5)", "v": f"{g['correctness_mean']} (n={g['correctness_scored']})"},
            ], {"m": "Measure", "v": "Result"}),
            "",
            "Citation correctness by question type: "
            + ", ".join(f"{t} {v}%" for t, v in g["citation_by_type_pct"].items()),
            "",
            "![Answer quality](answer_quality.png)",
            "",
            "## 3. Latency and memory",
            "",
            md_table([
                {"s": "Retrieval (hybrid + re-ranker)", "m": g["retrieval_mean_ms"], "p": g["retrieval_p95_ms"]},
                {"s": "Generation (LLM, streamed)", "m": g["generation_mean_ms"], "p": g["generation_p95_ms"]},
            ], {"s": "Stage", "m": "Mean (ms)", "p": "p95 (ms)"}),
            "",
        ]
    else:
        parts += ["## 3. Latency and memory", ""]
    parts += [
        f"Peak RAM: this Python process {ram.peak_app_mb:,.0f} MB · Ollama {ram.peak_ollama_mb:,.0f} MB · "
        f"combined {ram.peak_total_mb:,.0f} MB",
        "",
        "![Latency](latency.png)",
        "",
        "## Notes",
        "",
        "- The judge is the same local LLM that writes the answers, so its scores are indicative only.",
        "- Per-question details: `retrieval_per_question.csv` and `generation_results.csv`.",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval, citations, answer quality and latency.")
    parser.add_argument("--limit", type=int, help="Only use the first N questions (quick check)")
    parser.add_argument("--skip-generation", action="store_true", help="Only evaluate retrieval (fast)")
    parser.add_argument("--rebuild-index", action="store_true", help="Delete and rebuild the evaluation index")
    parser.add_argument("--no-figures", action="store_true", help="Index without vision-model captions (faster)")
    parser.add_argument("--k", type=int, default=5, help="Cut-off for Hit@k, Recall@k and MRR (default 5)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    set_console_level("WARNING")

    from evaluation.charts import make_charts
    from src.generation.llm import check_llm, get_llm
    from src.generation.prompts import JUDGE_PROMPT
    from src.generation.rag_chain import RAGChain
    from src.retrieval.hybrid_retriever import HybridRetriever
    from langchain_core.output_parsers import StrOutputParser

    documents, questions = load_eval_set()
    if args.limit:
        questions = questions[: args.limit]
    settings = eval_settings(args.no_figures)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_generation:
        problem = check_llm(settings)
        if problem:
            sys.exit(f"ERROR: {problem}\n(Use --skip-generation to evaluate retrieval only.)")

    with RamMonitor() as ram:
        print("1/4 Building the evaluation index")
        indexer = build_index(settings, documents, args.rebuild_index)
        retriever = HybridRetriever(indexer.vector_store, indexer.bm25, settings=settings)
        print("    loading models...")
        retriever.warm_up()

        print(f"2/4 Retrieval: {len(questions)} questions x 4 set-ups")
        retrieval_rows = evaluate_retrieval(retriever, questions, args.k)

        generation_rows, generation = [], None
        if not args.skip_generation:
            print(f"3/4 Answers + LLM judge: {len(questions)} questions")
            chain = RAGChain(retriever, settings)
            judge_chain = JUDGE_PROMPT | get_llm(settings, temperature=0.0) | StrOutputParser()
            generation_rows = evaluate_generation(chain, judge_chain, questions)
            generation = summarize_generation(generation_rows)
        else:
            print("3/4 Answers skipped (--skip-generation)")

    print("4/4 Writing the report")
    retrieval = summarize_retrieval(retrieval_rows)
    by_type = summarize_by_type(retrieval_rows)
    write_csv(REPORTS_DIR / "retrieval_metrics.csv", retrieval)
    write_csv(REPORTS_DIR / "retrieval_by_type.csv", by_type)
    write_csv(REPORTS_DIR / "retrieval_per_question.csv", retrieval_rows)
    write_csv(REPORTS_DIR / "generation_results.csv", generation_rows)

    memory = psutil.virtual_memory().total / 1e9
    info = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "questions": len(questions), "documents": len(documents),
        "k": args.k, "llm": settings.active_llm_model, "embeddings": settings.active_embedding_model,
        "reranker": retriever.reranker.model_name, "vision": settings.active_vision_model
        if settings.caption_figures else "off",
        "machine": f"{platform.processor() or platform.machine()}, {psutil.cpu_count()} logical CPUs, "
                   f"{memory:.0f} GB RAM, {platform.system()} {platform.release()}",
    }
    write_summary_md(REPORTS_DIR / "summary.md", info, retrieval, by_type, generation, ram)
    make_charts(REPORTS_DIR, retrieval, by_type, generation)

    print(f"\nRetrieval (k={args.k}):")
    for r in retrieval:
        print(f"  {r['setup']:<20} Hit@5 {r['hit_at_5']:.3f}  Recall@5 {r['recall_at_5']:.3f}  "
              f"MRR {r['mrr']:.3f}  mean {r['mean_latency_ms']:.0f} ms")
    if generation:
        print(f"\nCitation correctness {generation['citation_correct_pct']}% | faithfulness "
              f"{generation['faithfulness_mean']} | correctness {generation['correctness_mean']} | "
              f"generation mean {generation['generation_mean_ms'] / 1000:.1f} s")
    print(f"Peak RAM: app {ram.peak_app_mb:,.0f} MB, Ollama {ram.peak_ollama_mb:,.0f} MB")
    print(f"\nReport written to {REPORTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
