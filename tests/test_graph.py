"""Graph-wiring tests: no real Claude/Chroma calls.

Monkeypatches src.graph.call_llm and swaps in a fake vectorstore so the
LangGraph state machine (retrieve -> grade -> rewrite/answer) is verified
in isolation from any network or API key requirement.
"""
import json

import pytest
from langchain_core.documents import Document

import src.graph as graph_mod
from src.graph import RetrievalGraph, _normalize_law_number


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3", "3"),
        ("Law 3", "3"),
        ("Article 15", "15"),
        ("law no. 12", "12"),
        ("Section 4", "4"),
    ],
)
def test_normalize_law_number_strips_non_numeric_formatting(raw, expected):
    assert _normalize_law_number(raw) == expected


class FakeVectorStore:
    def __init__(self, results_by_query: dict[str, list[tuple[Document, float]]]):
        self.results_by_query = results_by_query
        self.queries_seen: list[str] = []

    def similarity_search_with_score(self, query, k=5):
        self.queries_seen.append(query)
        return self.results_by_query.get(query, [])


def _doc(law_number, section_title, text, source_doc="laws.pdf"):
    return Document(
        page_content=text,
        metadata={"law_number": law_number, "law_title": "", "section_title": section_title, "source_doc": source_doc},
    )


def test_answers_directly_when_first_retrieval_is_relevant(monkeypatch):
    store = FakeVectorStore(
        {"how many players?": [(_doc("3", "Number of Players", "A team has eleven players."), 0.1)]}
    )

    def fake_call(system, user, model, max_tokens=1024):
        if "relevance grader" in system:
            return json.dumps({"relevant": [True]})
        return json.dumps({"answer": "Eleven players per team (Law 3).", "cited_laws": ["3"], "answerable": True})

    monkeypatch.setattr(graph_mod, "call_llm", fake_call)

    rg = RetrievalGraph(store)
    result = rg.query("how many players?")

    assert result["answerable"] is True
    assert result["cited_laws"] == ["3"]
    assert store.queries_seen == ["how many players?"]  # no retry needed


def test_rewrites_and_retries_once_when_first_pass_finds_nothing(monkeypatch):
    store = FakeVectorStore(
        {
            "ambiguous q": [(_doc("1", "Irrelevant", "unrelated text"), 0.9)],
            "rewritten q": [(_doc("11", "Offside", "A player is offside if..."), 0.1)],
        }
    )

    call_log = []

    def fake_call(system, user, model, max_tokens=1024):
        call_log.append(system[:20])
        if "rewrite football" in system:
            return json.dumps({"rewritten_question": "rewritten q"})
        if "relevance grader" in system:
            relevant = "Offside" in user
            return json.dumps({"relevant": [relevant]})
        return json.dumps({"answer": "Offside explanation (Law 11).", "cited_laws": ["11"], "answerable": True})

    monkeypatch.setattr(graph_mod, "call_llm", fake_call)

    rg = RetrievalGraph(store)
    result = rg.query("ambiguous q")

    assert store.queries_seen == ["ambiguous q", "rewritten q"]
    assert result["cited_laws"] == ["11"]
    assert result["retry_count"] == 1


def test_says_i_dont_know_when_nothing_relevant_even_after_retry(monkeypatch):
    store = FakeVectorStore(
        {
            "no coverage question": [(_doc("1", "Field", "field dimensions"), 0.9)],
            "rewritten no coverage": [(_doc("1", "Field", "field dimensions"), 0.9)],
        }
    )

    def fake_call(system, user, model, max_tokens=1024):
        if "rewrites football" in system:
            return json.dumps({"rewritten_question": "rewritten no coverage"})
        if "relevance grader" in system:
            return json.dumps({"relevant": [False]})
        raise AssertionError("answer generation should not be called with no relevant chunks")

    monkeypatch.setattr(graph_mod, "call_llm", fake_call)

    rg = RetrievalGraph(store)
    result = rg.query("no coverage question")

    assert result["answerable"] is False
    assert result["cited_laws"] == []
    assert "don't know" in result["answer"].lower()
    assert result["retry_count"] == 1  # retried exactly once, not looped forever


def test_grading_api_failure_is_flagged_distinctly_from_a_real_refusal(monkeypatch):
    # This is the exact failure mode that hit a real Gemini free-tier quota
    # exhaustion during manual testing: every grading call raised, and
    # without grading_error, the report would have shown a plain
    # "over_refusal" indistinguishable from the model genuinely deciding
    # nothing was relevant.
    store = FakeVectorStore(
        {"q": [(_doc("3", "Number of Players", "A team has eleven players."), 0.1)]}
    )

    def fake_call(system, user, model, max_tokens=1024):
        if "relevance grader" in system:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        raise AssertionError("answer generation should not be called with no relevant chunks")

    monkeypatch.setattr(graph_mod, "call_llm", fake_call)

    rg = RetrievalGraph(store)
    result = rg.query("q")

    assert result["answerable"] is False
    assert result["grading_error"] is True


def test_retrieved_state_carries_the_real_grading_verdict_for_every_chunk(monkeypatch):
    # Regression test: _grade used to mutate the "relevant" flag only on
    # the filtered relevant_chunks list, leaving state["retrieved"] (the
    # full ranked list the web app's /api/query exposes for transparency)
    # permanently stuck at relevant=None for every chunk, graded or not.
    store = FakeVectorStore(
        {
            "q": [
                (_doc("3", "Number of Players", "eleven players"), 0.9),
                (_doc("9", "Ball in play", "unrelated text"), 0.8),
            ]
        }
    )

    def fake_call(system, user, model, max_tokens=1024):
        if "relevance grader" in system:
            return json.dumps({"relevant": [True, False]})
        return json.dumps({"answer": "Eleven (Law 3).", "cited_laws": ["3"], "answerable": True})

    monkeypatch.setattr(graph_mod, "call_llm", fake_call)

    rg = RetrievalGraph(store)
    result = rg.query("q")

    assert [c["relevant"] for c in result["retrieved"]] == [True, False]
