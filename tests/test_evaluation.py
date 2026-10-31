"""Phase 8 tests: evaluation metrics, judge parsing, and the eval set itself."""

import pytest

from evaluation.run_eval import (
    EVAL_SET,
    hit_at_k,
    load_eval_set,
    p95,
    parse_judge_scores,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
)

RETRIEVED = [("a.pdf", 3), ("b.pdf", 2), ("a.pdf", 2), ("a.pdf", 7), ("a.pdf", 5), ("a.pdf", 9)]


def test_relevance_needs_right_file_and_page():
    assert relevance_flags(RETRIEVED, "a.pdf", [2]) == [False, False, True, False, False, False]


def test_hit_at_k():
    flags = relevance_flags(RETRIEVED, "a.pdf", [2])
    assert hit_at_k(flags, 5) == 1.0
    assert hit_at_k(flags, 2) == 0.0


def test_recall_counts_expected_pages_found():
    assert recall_at_k(RETRIEVED, "a.pdf", [2, 9], k=5) == 0.5  # page 9 is only at rank 6
    assert recall_at_k(RETRIEVED, "a.pdf", [2, 9], k=6) == 1.0
    assert recall_at_k(RETRIEVED, "c.pdf", [1], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(relevance_flags(RETRIEVED, "a.pdf", [2]), 5) == pytest.approx(1 / 3)
    assert reciprocal_rank(relevance_flags(RETRIEVED, "a.pdf", [3]), 5) == 1.0
    assert reciprocal_rank(relevance_flags(RETRIEVED, "a.pdf", [9]), 5) == 0.0  # outside top 5


def test_p95_nearest_rank():
    assert p95(list(range(1, 101))) == 95
    assert p95([5.0]) == 5.0
    assert p95([]) == 0.0


def test_parse_judge_scores():
    assert parse_judge_scores("FAITHFULNESS: 4\nCORRECTNESS: 5") == (4, 5)
    assert parse_judge_scores("**Faithfulness**: 3 / 5\ncorrectness - 2") == (3, 2)
    assert parse_judge_scores("I think it is good.") == (None, None)


def test_eval_set_is_complete():
    documents, questions = load_eval_set(EVAL_SET)
    assert len(questions) >= 25
    assert len(documents) >= 2
    assert {q.type for q in questions} == {"text", "table", "figure", "ocr"}
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.expected_file in documents
        assert q.expected_pages and all(p >= 1 for p in q.expected_pages)
        assert q.question.strip() and q.expected_answer.strip()
