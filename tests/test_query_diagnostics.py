"""Unit tests for src/query_diagnostics.py.

The scoring math (rank_chunks) is pure and needs no mocking -- a fake
vectorstore stands in for Chroma/InMemoryVectorStore. Only the LLM call
inside suggest_better_phrasing is mocked; the re-scoring of its candidate
rephrasings still runs the real local math against the fake vectorstore.
"""
import json

from langchain_core.documents import Document

import src.query_diagnostics as diagnostics


class FakeVectorStore:
    """Returns canned (doc, score) results per exact query string, so tests
    can control the "retrieval math" outcome without a real embedding model."""

    def __init__(self, results_by_query: dict[str, list[tuple[Document, float]]]):
        self.results_by_query = results_by_query

    def similarity_search_with_score(self, query, k=5):
        return self.results_by_query.get(query, [])[:k]


def _doc(law_number, section_title):
    return Document(page_content="text", metadata={"law_number": law_number, "section_title": section_title})


def test_rank_chunks_reports_real_scores_in_order():
    store = FakeVectorStore({"q": [(_doc("11", "Offside position"), 0.98), (_doc("9", "Ball in play"), 0.90)]})

    ranking = diagnostics.rank_chunks(store, "q", top_k=5)

    assert ranking == [
        {"law_number": "11", "section_title": "Offside position", "score": 0.98},
        {"law_number": "9", "section_title": "Ball in play", "score": 0.90},
    ]


def test_suggest_better_phrasing_returns_none_when_no_candidate_improves_meaningfully(monkeypatch):
    store = FakeVectorStore(
        {
            "What is an offside?": [(_doc("9", "Ball in play"), 0.90)],
            "rephrased a": [(_doc("9", "Ball in play"), 0.905)],  # within noise margin
        }
    )
    monkeypatch.setattr(
        diagnostics, "call_llm", lambda *a, **k: json.dumps({"rephrasings": ["rephrased a"]})
    )

    result = diagnostics.suggest_better_phrasing(store, "What is an offside?", "some-model")

    assert result is None


def test_suggest_better_phrasing_returns_the_best_candidate_that_clears_the_bar(monkeypatch):
    store = FakeVectorStore(
        {
            "What is an offside?": [(_doc("9", "Ball in play"), 0.808)],
            "candidate A": [(_doc("9", "Ball in play"), 0.81)],  # barely better -- not enough
            "candidate B": [(_doc("11", "Offside position"), 0.982)],  # real improvement
        }
    )
    monkeypatch.setattr(
        diagnostics,
        "call_llm",
        lambda *a, **k: json.dumps({"rephrasings": ["candidate A", "candidate B"]}),
    )

    result = diagnostics.suggest_better_phrasing(store, "What is an offside?", "some-model")

    assert result is not None
    assert result["suggested_phrasing"] == "candidate B"
    assert result["suggested_top_score"] == 0.982
    assert result["original_top_score"] == 0.808
    assert result["suggested_ranking"][0]["law_number"] == "11"


def test_suggest_better_phrasing_returns_none_on_llm_failure(monkeypatch):
    store = FakeVectorStore({"q": [(_doc("1", "x"), 0.9)]})

    def raise_error(*a, **k):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(diagnostics, "call_llm", raise_error)

    assert diagnostics.suggest_better_phrasing(store, "q", "some-model") is None


def test_suggest_better_phrasing_ignores_malformed_candidates(monkeypatch):
    store = FakeVectorStore(
        {
            "q": [(_doc("9", "x"), 0.80)],
            "good rephrasing": [(_doc("1", "y"), 0.95)],
        }
    )
    monkeypatch.setattr(
        diagnostics,
        "call_llm",
        lambda *a, **k: json.dumps({"rephrasings": ["", 42, None, "good rephrasing"]}),
    )

    result = diagnostics.suggest_better_phrasing(store, "q", "some-model")

    assert result is not None
    assert result["suggested_phrasing"] == "good rephrasing"
