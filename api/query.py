"""Vercel serverless function backing the deployed web demo.

A single Flask (WSGI) app handling both routes this project needs:
  GET  /api/status  -> quota tracker state + which embedding backends are live
  POST /api/query   -> {"question": "...", "embedding_model": "spacy"|"minilm"} -> the RAG answer

Everything here reuses the same src/graph.py LangGraph pipeline the CLI and
tests use. The one thing that's different in this deployment is the
vector store: Chroma persists to local disk, which doesn't survive on
Vercel's ephemeral, non-shared serverless filesystem, so this loads the
tiny precomputed in-memory store instead (src/memory_vectorstore.py) --
see that module's docstring and the README's "Deploying" section for why.

Two embedding backends are wired up so users can directly compare
retrieval quality (see src/embeddings.py): "spacy" (averaged word
vectors, always available -- this is the original default) and "minilm"
(a real sentence-transformer, run via fastembed/ONNX -- see
SentenceTransformerEmbeddings' docstring for why it needs its own
precomputed data/processed/embeddings_minilm.npy artifact, which is not
guaranteed to exist in every deployment). The LLM backend (Claude/Gemini,
via src/llm.py) is deliberately kept IDENTICAL across both -- the only
variable a user is comparing here is the embedding model, not the LLM.

Each backend's graph is built once at module scope so a warm container
reuses it across requests instead of reloading spaCy/fastembed + the
embedding matrix on every single call; only a cold start pays that cost.
The minilm graph is built lazily, on first request that asks for it, so a
deployment that never populated embeddings_minilm.npy doesn't pay for
loading it (or crash at import time) when no one uses it.
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

CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.json"

EMBEDDING_MODELS = {
    "spacy": {
        "label": "spaCy word vectors (baseline)",
        "embeddings_path": REPO_ROOT / "data" / "processed" / "embeddings_spacy.npy",
    },
    "minilm": {
        "label": "MiniLM sentence-transformer (real semantic embeddings)",
        "embeddings_path": REPO_ROOT / "data" / "processed" / "embeddings_minilm.npy",
    },
}
DEFAULT_EMBEDDING_MODEL = "spacy"

# Lazily populated per key ("spacy" | "minilm") the first time each is
# actually requested -- see module docstring for why minilm isn't built
# eagerly at import time.
_graphs: dict[str, RetrievalGraph] = {}
_graph_errors: dict[str, str] = {}


def _get_graph(embedding_model: str) -> RetrievalGraph:
    if embedding_model in _graphs:
        return _graphs[embedding_model]
    if embedding_model in _graph_errors:
        raise RuntimeError(_graph_errors[embedding_model])

    config = EMBEDDING_MODELS[embedding_model]
    if not config["embeddings_path"].exists():
        msg = (
            f"'{embedding_model}' embedding backend isn't set up on this deployment "
            f"({config['embeddings_path'].name} is missing -- run "
            f"`python -m scripts.build_embeddings_artifact --backend {embedding_model}` "
            "and redeploy)."
        )
        _graph_errors[embedding_model] = msg
        raise RuntimeError(msg)

    store = InMemoryVectorStore(
        chunks_path=CHUNKS_PATH,
        embeddings_path=config["embeddings_path"],
        embeddings_backend=get_embeddings(embedding_model),
    )
    graph = RetrievalGraph(store)
    _graphs[embedding_model] = graph
    return graph


# Build the default (spacy) graph eagerly, same as before this feature --
# it's always available and every request needs it available fast.
_get_graph(DEFAULT_EMBEDDING_MODEL)


@app.get("/api/status")
def status():
    status_data = quota_store.get_status()
    status_data["embedding_models"] = {
        key: {"label": cfg["label"], "available": cfg["embeddings_path"].exists()}
        for key, cfg in EMBEDDING_MODELS.items()
    }
    return jsonify(status_data)


@app.post("/api/query")
def query():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 500:
        return jsonify({"error": "question is too long (500 char max)"}), 400

    embedding_model = (body.get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip().lower()
    if embedding_model not in EMBEDDING_MODELS:
        return jsonify(
            {"error": f"embedding_model must be one of {sorted(EMBEDDING_MODELS)}"}
        ), 400
    try:
        graph = _get_graph(embedding_model)
    except RuntimeError as e:
        return jsonify({"error": "embedding_model_unavailable", "message": str(e)}), 503

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

    result = graph.query(question)

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

    # grading_error/generation_error mean an underlying call broke (after
    # retries -- see src/llm.py's 503 handling) and the pipeline failed
    # closed, NOT that it genuinely searched the corpus and found nothing.
    # Without this check, both cases render as the identical "I don't
    # know," which reads to a user as "this tool doesn't work" when the
    # real story is "Gemini had a bad moment, try again." Distinguishing
    # them is a users'-trust issue, not a cosmetic one.
    infra_error = bool(result.get("grading_error") or result.get("generation_error"))
    if not result["answerable"] and infra_error:
        return jsonify(
            {
                "error": "temporarily_unavailable",
                "message": "Google Gemini appears to be down or overloaded right now. Please try again in a moment.",
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
        "embedding_model": embedding_model,
    }

    # Only spend the extra LLM call generating/re-scoring alternate
    # phrasings when the pipeline genuinely searched and found nothing --
    # suggesting a "better phrasing" makes no sense when the real cause
    # was an outage (checked above), and would likely fail the same way.
    # Re-scored against the SAME embedding backend the request asked for,
    # so a suggestion is never phrased in terms of a different model's math.
    if not result["answerable"]:
        suggestion = suggest_better_phrasing(graph.vectorstore, question, DEFAULT_MODEL)
        if suggestion:
            response["suggestion"] = suggestion

    return jsonify(response)
