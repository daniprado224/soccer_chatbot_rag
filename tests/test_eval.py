from src.eval import (
    QuestionResult,
    _classify_failure,
    precision_at_k,
    recall_at_k,
    render_markdown_report,
)


def test_precision_at_k_counts_hits_over_k():
    assert precision_at_k(["3", "3", "11", "1", "3"], {"3"}) == 3 / 5


def test_precision_at_k_empty_retrieval_is_zero():
    assert precision_at_k([], {"3"}) == 0.0


def test_recall_at_k_finds_expected_law_anywhere_in_topk():
    assert recall_at_k(["1", "3", "11"], {"3"}) == 1.0
    assert recall_at_k(["1", "2", "9"], {"3"}) == 0.0


def test_recall_at_k_undefined_for_no_expected_laws():
    assert recall_at_k(["1", "2"], set()) is None


def _base_result(**overrides) -> QuestionResult:
    defaults = dict(
        id="q1",
        category="easy",
        question="q",
        expected_law_numbers=["3"],
        expected_answerable=True,
        retrieved_law_numbers=["3", "1"],
        precision=0.5,
        recall=1.0,
        cited_laws=["3"],
        answer="eleven players",
        answerable=True,
        retry_count=0,
        citation_overlap=True,
        citation_exact=True,
        answerability_correct=True,
        failure_mode="",
    )
    defaults.update(overrides)
    return QuestionResult(**defaults)


def test_classify_failure_passing_case():
    r = _base_result()
    assert _classify_failure(r) == "none"


def test_classify_failure_retrieval_failure_when_recall_zero():
    r = _base_result(recall=0.0, citation_overlap=False, citation_exact=False)
    assert _classify_failure(r) == "retrieval"


def test_classify_failure_generation_failure_when_retrieved_but_not_cited():
    r = _base_result(recall=1.0, citation_overlap=False, citation_exact=False)
    assert _classify_failure(r) == "generation"


def test_classify_failure_over_refusal():
    r = _base_result(answerable=False, answerability_correct=False, citation_overlap=False)
    assert _classify_failure(r) == "over_refusal"


def test_classify_failure_hallucination_on_unanswerable_question():
    r = _base_result(
        expected_law_numbers=[],
        expected_answerable=False,
        precision=None,
        recall=None,
        citation_overlap=None,
        citation_exact=None,
        answerable=True,
        answerability_correct=False,
    )
    assert _classify_failure(r) == "hallucination"


def test_classify_failure_correct_refusal_on_unanswerable_question():
    r = _base_result(
        expected_law_numbers=[],
        expected_answerable=False,
        precision=None,
        recall=None,
        citation_overlap=None,
        citation_exact=None,
        cited_laws=[],
        answerable=False,
        answerability_correct=True,
    )
    assert _classify_failure(r) == "none"


def test_render_markdown_report_includes_key_sections():
    results = [
        _base_result(),
        _base_result(
            id="q2",
            category="unanswerable",
            expected_law_numbers=[],
            expected_answerable=False,
            precision=None,
            recall=None,
            cited_laws=[],
            answerable=False,
            citation_overlap=None,
            citation_exact=None,
            answerability_correct=True,
            failure_mode="none",
        ),
    ]
    for r in results:
        r.failure_mode = _classify_failure(r)

    report = render_markdown_report(results, top_k=5)

    assert "# RAG Evaluation Report" in report
    assert "Retrieval metrics" in report
    assert "Generation metrics" in report
    assert "Failure attribution" in report
    assert "q1" in report and "q2" in report
