"""Retrieval-score transparency for the web demo: expose the actual ranked
chunks a query retrieved, and -- only when the pipeline fails to answer --
suggest an alternate phrasing that scores measurably higher, re-scored
against the exact same math as the real retriever.

The scoring itself is pure, deterministic, LLM-free math: every chunk and
every query is embedded as an L2-normalized averaged spaCy word vector, so
a dot product between two normalized vectors IS their cosine similarity.
See README's "How retrieval scoring works" section for the full
explanation and a worked example. The only LLM involvement here is
generating CANDIDATE rephrasings to then score with that same math -- the
suggestion is never accepted on the LLM's say-so, only if it demonstrably
scores higher against the real retriever.
"""
from __future__ import annotations

from .llm import call_llm, extract_json

# Only worth spending an extra LLM call on phrasing suggestions when the
# pipeline actually failed to answer -- when it already worked, there's
# nothing to suggest, and this keeps the feature's cost bounded to
# genuine failures rather than every single question.
NUM_CANDIDATES = 3
# A suggested phrasing must beat the original by more than noise -- these
# averaged-word-vector scores cluster tightly (typically 0.80-0.98
# regardless of match quality, since most rulebook sentences share a lot
# of common function-word vocabulary), so a small raw delta is often not
# meaningful. 3% relative improvement is a judgment call, not a
# statistically derived cutoff -- see README.
MIN_RELATIVE_IMPROVEMENT = 1.03


def rank_chunks(vectorstore, question: str, top_k: int = 5) -> list[dict]:
    """The real, already-computed retrieval ranking for a question -- no
    extra cost, this is exactly what the answer pipeline itself used."""
    hits = vectorstore.similarity_search_with_score(question, k=top_k)
    return [
        {
            "law_number": doc.metadata.get("law_number", ""),
            "section_title": doc.metadata.get("section_title", ""),
            "score": round(float(score), 4),
        }
        for doc, score in hits
    ]


def suggest_better_phrasing(
    vectorstore, question: str, model: str, top_k: int = 5
) -> dict | None:
    """Generate alternate phrasings and return the one that measurably
    retrieves more confidently, or None if none clears the bar.

    Every number in the returned dict comes from the same
    similarity_search_with_score() cosine-similarity computation the real
    retriever uses -- nothing here is estimated or LLM-judged.
    """
    original_ranking = rank_chunks(vectorstore, question, top_k=top_k)
    original_top_score = original_ranking[0]["score"] if original_ranking else 0.0

    system = (
        f"Rewrite the following football (soccer) rules question in "
        f"{NUM_CANDIDATES} different ways. Vary vocabulary and sentence "
        "structure while preserving the exact meaning -- try phrasing it "
        "closer to how an official rulebook would state the underlying "
        'rule. Respond with ONLY JSON: {"rephrasings": ["...", "...", "..."]}'
    )
    user = f"Question: {question}"
    try:
        raw = call_llm(system, user, model, max_tokens=300)
        rephrasings = extract_json(raw).get("rephrasings", [])
    except Exception:
        return None  # a diagnostic feature failing must never break the real answer

    best = None
    for phrasing in rephrasings:
        if not isinstance(phrasing, str) or not phrasing.strip():
            continue
        ranking = rank_chunks(vectorstore, phrasing, top_k=top_k)
        top_score = ranking[0]["score"] if ranking else 0.0
        if best is None or top_score > best["top_score"]:
            best = {"phrasing": phrasing, "top_score": top_score, "ranking": ranking}

    if best is None or best["top_score"] <= original_top_score * MIN_RELATIVE_IMPROVEMENT:
        return None

    return {
        "original_top_score": original_top_score,
        "suggested_phrasing": best["phrasing"],
        "suggested_top_score": best["top_score"],
        "suggested_ranking": best["ranking"],
    }
