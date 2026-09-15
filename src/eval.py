"""Evaluation harness.

Reports retrieval quality (precision@k / recall@k against ground-truth Law
numbers) SEPARATELY from generation quality (does the final answer cite the
right law, and does it correctly refuse when the corpus has no answer), so
a failure can be attributed to "retrieval never found it" vs "retrieval
found it but generation didn't use/cite it correctly".

Pure metric functions and the markdown renderer take plain data and are
unit tested (tests/test_eval.py) without any API key or vector store.
``run_eval`` is the only piece that needs a live RetrievalGraph.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QA_PATH = REPO_ROOT / "data" / "eval" / "qa_pairs.json"


def precision_at_k(retrieved_law_numbers: list[str], expected: set[str]) -> float:
    """Fraction of the k retrieved chunks whose law_number is in the expected set."""
    if not retrieved_law_numbers:
        return 0.0
    hits = sum(1 for law in retrieved_law_numbers if law in expected)
    return hits / len(retrieved_law_numbers)


def recall_at_k(retrieved_law_numbers: list[str], expected: set[str]) -> float | None:
    """Fraction of expected law numbers that appear anywhere in the k retrieved chunks.

    Returns None (undefined, not zero) when there is no expected law -- i.e.
    for questions the corpus should not answer at all.
    """
    if not expected:
        return None
    hits = len(set(retrieved_law_numbers) & expected)
    return hits / len(expected)


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    expected_law_numbers: list[str]
    expected_answerable: bool
    retrieved_law_numbers: list[str]
    precision: float | None
    recall: float | None
    cited_laws: list[str]
    answer: str
    answerable: bool
    retry_count: int
    citation_overlap: bool | None  # any overlap between cited and expected laws
    citation_exact: bool | None  # exact set match
    answerability_correct: bool
    failure_mode: str  # "none" | "retrieval" | "generation" | "hallucination" | "over_refusal"


def _classify_failure(r: "QuestionResult") -> str:
    if r.expected_answerable:
        if r.answerability_correct and r.citation_overlap:
            return "none"
        if not r.answerable:
            return "over_refusal"
        # answered, but wrong/incomplete citation -- was the law ever retrieved?
        if r.recall is not None and r.recall == 0.0:
            return "retrieval"
        return "generation"
    else:
        return "none" if r.answerability_correct else "hallucination"


def run_eval(graph, qa_pairs: list[dict], top_k: int) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for qa in qa_pairs:
        expected_laws = set(qa["expected_law_numbers"] or [])

        raw_hits = graph.vectorstore.similarity_search_with_score(qa["question"], k=top_k)
        retrieved_laws = [doc.metadata.get("law_number", "") for doc, _ in raw_hits]

        graph_result = graph.query(qa["question"])

        cited = set(graph_result["cited_laws"])
        precision = precision_at_k(retrieved_laws, expected_laws) if qa["expected_answerable"] else None
        recall = recall_at_k(retrieved_laws, expected_laws) if qa["expected_answerable"] else None
        citation_overlap = bool(cited & expected_laws) if qa["expected_answerable"] else None
        citation_exact = (cited == expected_laws) if qa["expected_answerable"] else None
        answerability_correct = graph_result["answerable"] == qa["expected_answerable"]

        r = QuestionResult(
            id=qa["id"],
            category=qa["category"],
            question=qa["question"],
            expected_law_numbers=sorted(expected_laws),
            expected_answerable=qa["expected_answerable"],
            retrieved_law_numbers=retrieved_laws,
            precision=precision,
            recall=recall,
            cited_laws=sorted(cited),
            answer=graph_result["answer"],
            answerable=graph_result["answerable"],
            retry_count=graph_result["retry_count"],
            citation_overlap=citation_overlap,
            citation_exact=citation_exact,
            answerability_correct=answerability_correct,
            failure_mode="",
        )
        r.failure_mode = _classify_failure(r)
        results.append(r)
    return results


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def render_markdown_report(results: list[QuestionResult], top_k: int) -> str:
    categories = sorted(set(r.category for r in results))
    lines: list[str] = []
    lines.append("# RAG Evaluation Report\n")
    lines.append(f"Retrieval depth: top-{top_k}. {len(results)} questions.\n")

    lines.append("## Retrieval metrics (answerable questions only)\n")
    lines.append("| Category | n | Precision@k | Recall@k |")
    lines.append("|---|---|---|---|")
    for cat in categories:
        rows = [r for r in results if r.category == cat and r.expected_answerable]
        if not rows:
            continue
        lines.append(
            f"| {cat} | {len(rows)} | {_fmt(_mean([r.precision for r in rows]))} "
            f"| {_fmt(_mean([r.recall for r in rows]))} |"
        )
    answerable = [r for r in results if r.expected_answerable]
    lines.append(
        f"| **overall** | {len(answerable)} | {_fmt(_mean([r.precision for r in answerable]))} "
        f"| {_fmt(_mean([r.recall for r in answerable]))} |\n"
    )

    lines.append("## Generation metrics (separate from retrieval)\n")
    lines.append("| Category | n | Citation overlap | Citation exact | Answerability correct |")
    lines.append("|---|---|---|---|---|")
    for cat in categories:
        rows = [r for r in results if r.category == cat]
        ans_rows = [r for r in rows if r.expected_answerable]
        lines.append(
            f"| {cat} | {len(rows)} "
            f"| {_fmt(_mean([1.0 if r.citation_overlap else 0.0 for r in ans_rows]))} "
            f"| {_fmt(_mean([1.0 if r.citation_exact else 0.0 for r in ans_rows]))} "
            f"| {_fmt(_mean([1.0 if r.answerability_correct else 0.0 for r in rows]))} |"
        )
    lines.append(
        f"| **overall** | {len(results)} "
        f"| {_fmt(_mean([1.0 if r.citation_overlap else 0.0 for r in answerable]))} "
        f"| {_fmt(_mean([1.0 if r.citation_exact else 0.0 for r in answerable]))} "
        f"| {_fmt(_mean([1.0 if r.answerability_correct else 0.0 for r in results]))} |\n"
    )

    lines.append("## Failure attribution\n")
    lines.append(
        "Every non-passing question is bucketed by where in the pipeline it broke, "
        "so a low score can be traced to retrieval or generation:\n"
    )
    lines.append("- `retrieval`: the correct law was never in the top-k at all (recall@k = 0)")
    lines.append("- `generation`: the correct law WAS retrieved, but the final answer didn't cite it")
    lines.append("- `over_refusal`: an answerable question was refused (\"I don't know\")")
    lines.append("- `hallucination`: an out-of-corpus question was answered instead of refused\n")
    lines.append("| Failure mode | count |")
    lines.append("|---|---|")
    for mode in ["retrieval", "generation", "over_refusal", "hallucination"]:
        n = sum(1 for r in results if r.failure_mode == mode)
        if n:
            lines.append(f"| {mode} | {n} |")
    passing = sum(1 for r in results if r.failure_mode == "none")
    lines.append(f"| none (passing) | {passing} |\n")

    lines.append("## Per-question detail\n")
    lines.append("| id | category | expected law(s) | cited law(s) | recall@k | answerable (exp/got) | failure |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        exp = ",".join(r.expected_law_numbers) or "-"
        cit = ",".join(r.cited_laws) or "-"
        lines.append(
            f"| {r.id} | {r.category} | {exp} | {cit} | {_fmt(r.recall)} "
            f"| {r.expected_answerable}/{r.answerable} | {r.failure_mode} |"
        )

    return "\n".join(lines) + "\n"


def load_qa_pairs(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["questions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--top-k", type=int, default=None, help="Defaults to RETRIEVAL_TOP_K env var")
    parser.add_argument("--persist", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--out", default="eval_report.md")
    parser.add_argument("--json-out", default="eval_results.json")
    args = parser.parse_args()

    from .graph import TOP_K, load_graph

    top_k = args.top_k or TOP_K
    graph = load_graph(args.persist, args.collection)
    qa_pairs = load_qa_pairs(Path(args.qa))

    print(f"[eval] running {len(qa_pairs)} questions at top-{top_k} ...")
    results = run_eval(graph, qa_pairs, top_k)

    Path(args.json_out).write_text(json.dumps([asdict(r) for r in results], indent=2))
    report = render_markdown_report(results, top_k)
    Path(args.out).write_text(report)
    print(f"[eval] wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
