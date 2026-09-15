"""Vercel serverless function backing the deployed web demo.

A single Flask (WSGI) app handling both routes this project needs:
  GET  /api/status  -> quota tracker state for the header bar
  POST /api/query   -> {"question": "..."} -> the RAG answer

Everything here reuses the same src/graph.py LangGraph pipeline the CLI and
tests use. The one thing that's different in this deployment is the
vector store: Chroma persists to local disk, which doesn't survive on
Vercel's ephemeral, non-shared serverless filesystem, so this loads the
tiny precomputed in-memory store instead (src/memory_vectorstore.py) --
see that module's docstring and the README's "Deploying" section for why.

The graph is built once at module scope so a warm container reuses it
across requests instead of reloading spaCy + the embedding matrix on
every single call; only a cold start pays that cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, request  # noqa: E402

from src import quota_store  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.graph import RetrievalGraph  # noqa: E402
from src.llm import DEFAULT_MODEL  # noqa: E402
from src.memory_vectorstore import InMemoryVectorStore  # noqa: E402
from src.query_diagnostics import suggest_better_phrasing  # noqa: E402

app = Flask(__name__)

_store = InMemoryVectorStore(
    chunks_path=REPO_ROOT / "data" / "processed" / "chunks.json",
    embeddings_path=REPO_ROOT / "data" / "processed" / "embeddings.npy",
    embeddings_backend=get_embeddings(),
)
_graph = RetrievalGraph(_store)


@app.get("/api/status")
def status():
    return jsonify(quota_store.get_status())


@app.post("/api/query")
def query():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 500:
        return jsonify({"error": "question is too long (500 char max)"}), 400

    pre_check = quota_store.get_status()
    if not pre_check["available"]:
        return jsonify(
            {
                "error": "temporarily_unavailable",
                "message": "The demo has hit its free-tier request limit. Try again shortly.",
                "reset_at": pre_check["reset_at"],
                "scope": pre_check["scope"],
            }
        ), 503

    result = _graph.query(question)

    if result.get("quota_info"):
        quota_store.mark_exhausted(
            result["quota_info"]["retry_after_seconds"], result["quota_info"]["scope"]
        )
        return jsonify(
            {
                "error": "temporarily_unavailable",
                "message": "The demo just hit its free-tier request limit. Try again shortly.",
            }
        ), 503

    quota_store.record_question()

    # The real ranking the pipeline itself just used to decide this answer
    # -- free to expose, it's already-computed data, not an extra call.
    retrieval = [
        {
            "law_number": c["law_number"],
            "section_title": c["section_title"],
            "score": round(c["score"], 4),
            "relevant": c["relevant"],
        }
        for c in result["retrieved"]
    ]

    response = {
        "answer": result["answer"],
        "cited_laws": result["cited_laws"],
        "answerable": result["answerable"],
        "retry_count": result["retry_count"],
        "retrieval": retrieval,
    }

    # Only spend the extra LLM call generating/re-scoring alternate
    # phrasings when the pipeline actually failed to answer -- see
    # src/query_diagnostics.py for why, and why this can never itself
    # break the real answer if it fails.
    if not result["answerable"]:
        suggestion = suggest_better_phrasing(_store, question, DEFAULT_MODEL)
        if suggestion:
            response["suggestion"] = suggestion

    return jsonify(response)
