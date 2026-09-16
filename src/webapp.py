"""Shared Flask app factory for the deployed web demo's serverless functions.

Split across TWO separate Vercel functions -- api/query.py (spaCy backend)
and api/minilm/query.py (MiniLM backend) -- rather than one function
serving both, the way this originally shipped. Bundling spaCy +
en_core_web_md (~180MB) AND fastembed + onnxruntime (~90MB+) into the SAME
function pushed it over Vercel's 500MB function size limit even after
trimming everything trimmable (see git history: dropping the unused
`anthropic` SDK only bought back ~5MB of actual compressed bundle size,
nowhere near its ~17MB raw install size -- Vercel's limit is on the
compressed bundle, and installed-directory size is a poor predictor of
that). Since any single request only ever needs ONE embedding backend,
splitting into two functions -- each importing only the backend IT
serves -- means neither bundle pays for the other's weight, with wide
margin on both sides instead of a razor-thin one.

Both functions share this exact route/error-handling logic via
build_app() so behavior can't drift between them.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from src import quota_store
from src.embeddings import get_embeddings
from src.graph import RetrievalGraph
from src.llm import DEFAULT_MODEL
from src.memory_vectorstore import InMemoryVectorStore
from src.query_diagnostics import suggest_better_phrasing

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def _build_graph(embedding_model: str) -> RetrievalGraph:
    config = EMBEDDING_MODELS[embedding_model]
    if not config["embeddings_path"].exists():
        raise RuntimeError(
            f"'{embedding_model}' embedding backend isn't set up on this deployment "
            f"({config['embeddings_path'].name} is missing -- run "
            f"`python -m scripts.build_embeddings_artifact --backend {embedding_model}` "
            "and redeploy)."
        )
    store = InMemoryVectorStore(
        chunks_path=CHUNKS_PATH,
        embeddings_path=config["embeddings_path"],
        embeddings_backend=get_embeddings(embedding_model),
    )
    return RetrievalGraph(store)


def build_app(embedding_model: str, route_prefix: str = "/api") -> Flask:
    """Build a Flask app scoped to exactly ONE embedding backend.

    Each deployed function calls this with its own fixed model -- the
    backend selection happens at deploy/routing time (which URL a request
    hits), not at request time inside one shared function. `route_prefix`
    must match wherever vercel.json routes that URL, since Vercel's
    Python/WSGI adapter passes the real incoming path through to this app
    unchanged (it doesn't rewrite it to match the function file's own
    location) -- so the Flask route itself has to already expect it.
    """
    if embedding_model not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding_model: {embedding_model!r}")

    app = Flask(__name__)

    graph: RetrievalGraph | None = None
    graph_error: str | None = None
    try:
        graph = _build_graph(embedding_model)
    except RuntimeError as e:
        # Don't crash at import time -- a deployment that never built this
        # backend's artifact should still serve /api/status (and this
        # function's own clean 503, below) instead of 500ing on cold start.
        graph_error = str(e)

    @app.get(f"{route_prefix}/status")
    def status():
        status_data = quota_store.get_status()
        status_data["embedding_models"] = {
            key: {"label": cfg["label"], "available": cfg["embeddings_path"].exists()}
            for key, cfg in EMBEDDING_MODELS.items()
        }
        return jsonify(status_data)

    @app.post(f"{route_prefix}/query")
    def query():
        if graph is None:
            return jsonify({"error": "embedding_model_unavailable", "message": graph_error}), 503

        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        if len(question) > 500:
            return jsonify({"error": "question is too long (500 char max)"}), 400

        requested_model = (body.get("embedding_model") or embedding_model).strip().lower()
        if requested_model != embedding_model:
            return jsonify(
                {
                    "error": "wrong_endpoint",
                    "message": (
                        f"This endpoint only serves embedding_model={embedding_model!r}; "
                        f"got {requested_model!r}."
                    ),
                }
            ), 400

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

        # grading_error/generation_error mean an underlying call broke
        # (after retries -- see src/llm.py's 503 handling) and the
        # pipeline failed closed, NOT that it genuinely searched the
        # corpus and found nothing. Without this check, both cases render
        # as the identical "I don't know," which reads to a user as "this
        # tool doesn't work" when the real story is "Gemini had a bad
        # moment, try again." Distinguishing them is a users'-trust
        # issue, not a cosmetic one.
        infra_error = bool(result.get("grading_error") or result.get("generation_error"))
        if not result["answerable"] and infra_error:
            return jsonify(
                {
                    "error": "temporarily_unavailable",
                    "message": "Google Gemini appears to be down or overloaded right now. Please try again in a moment.",
                }
            ), 503

        quota_store.record_question()

        # The real ranking the pipeline itself just used to decide this
        # answer -- free to expose, it's already-computed data, not an
        # extra call.
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
        # phrasings when the pipeline genuinely searched and found
        # nothing -- suggesting a "better phrasing" makes no sense when
        # the real cause was an outage (checked above), and would likely
        # fail the same way. Re-scored against this SAME embedding
        # backend, so a suggestion is never phrased in terms of a
        # different model's math.
        if not result["answerable"]:
            suggestion = suggest_better_phrasing(graph.vectorstore, question, DEFAULT_MODEL)
            if suggestion:
                response["suggestion"] = suggestion

        return jsonify(response)

    return app
