"""Embedding backends behind a common LangChain-compatible interface.

Only ``spacy`` is actually exercised end-to-end by this repo's own ingestion
and eval runs: the dev sandbox this project was built in has network egress
blocked to both api.openai.com and huggingface.co, so OpenAI embeddings and
HuggingFace-hosted local models (including ``minilm`` below) could not be
downloaded or called from here. ``en_core_web_md`` ships as a plain pip
wheel from a GitHub release asset, which was reachable, so it is the
default. ``minilm`` is real, working code -- just not something this repo's
own sandbox could run and measure; see SentenceTransformerEmbeddings'
docstring.

The retriever (graph.py) only depends on the LangChain ``Embeddings``
interface (``embed_documents`` / ``embed_query``), so swapping backends
never touches retrieval or graph code -- only this file and the
``EMBEDDING_BACKEND`` env var.
"""
from __future__ import annotations

import os

import numpy as np
from langchain_core.embeddings import Embeddings

DEFAULT_SPACY_MODEL = "en_core_web_md"


class SpacyVectorEmbeddings(Embeddings):
    """Averaged spaCy word vectors (GloVe-style static vectors, 300-dim).

    Local, deterministic, zero API cost. Weaker than a real sentence
    embedding model at paraphrase/semantic similarity since it averages
    per-token vectors with no attention or sentence-level training --
    see the README eval section for the measured retrieval-quality impact.
    """

    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL):
        import spacy

        try:
            # Disable pipeline components we don't need; keeps embedding
            # calls fast since only the vocab vectors are used.
            self.nlp = spacy.load(
                model_name, disable=["ner", "parser", "tagger", "lemmatizer", "attribute_ruler"]
            )
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. Run: "
                f"python -m spacy download {model_name}"
            ) from e
        if self.nlp.vocab.vectors_length == 0:
            raise RuntimeError(
                f"spaCy model '{model_name}' has no word vectors (loaded a "
                "non-'md'/'lg' model?)."
            )
        self.model_name = model_name

    def _embed_one(self, text: str) -> list[float]:
        doc = self.nlp(text)
        vec = doc.vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class SentenceTransformerEmbeddings(Embeddings):
    """A real sentence-transformer model (all-MiniLM-L6-v2): trained
    end-to-end on sentence pairs with an attention mechanism, so it
    captures paraphrase/semantic similarity instead of rewarding
    vocabulary overlap the way SpacyVectorEmbeddings does. This is the
    backend "phase 2" adds so users can directly compare retrieval quality
    against the spaCy baseline in the deployed app.

    Implemented via ``fastembed`` (ONNX Runtime) rather than the
    ``sentence-transformers`` package itself: sentence-transformers pulls
    in PyTorch, which alone runs 200-800MB depending on wheel -- likely to
    blow past Vercel's serverless function size limit once stacked with
    spaCy, langchain, etc (this project has already hit real deployment
    size/config problems once -- see git history). fastembed runs the same
    all-MiniLM-L6-v2 weights (quantized, ONNX-exported) through
    onnxruntime instead: same architecture and training, ~1/3 the
    deployed size, no torch dependency. It is not bit-identical to the
    original PyTorch model (quantization + ONNX export introduce small
    numeric differences), but produces effectively the same embeddings.

    IMPORTANT, honestly: fastembed downloads the ~90MB ONNX model from
    Hugging Face on first use and caches it locally -- this class does
    NOT ship the weights. That download could not be exercised or
    verified from the sandbox this repo was built in (huggingface.co is
    blocked by that sandbox's egress policy -- confirmed, not assumed).
    It should work fine from a normal machine or from Vercel's own
    network, but that is untested by this repo's own CI/eval the way the
    spaCy backend is. Run `python -m scripts.build_embeddings_artifact
    --backend minilm` somewhere with real network access before trusting
    this backend's numbers.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "fastembed is not installed. Run: pip install fastembed"
            ) from e

        self.model_name = model_name or os.getenv("MINILM_MODEL", self.MODEL_NAME)
        # Vercel's Python runtime is Lambda-based: /tmp is the only writable
        # directory, so default there when running on Vercel (VERCEL=1 is
        # set automatically by their platform) rather than trying to write
        # next to the (read-only, in production) deployment bundle.
        cache_dir = cache_dir or os.getenv("FASTEMBED_CACHE_DIR")
        if not cache_dir and os.getenv("VERCEL"):
            cache_dir = "/tmp/fastembed_cache"
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        try:
            self._model = TextEmbedding(model_name=self.model_name, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Could not load sentence-transformer backend {self.model_name!r} via "
                f"fastembed. This requires network access to huggingface.co on first "
                f"use to download the ONNX model weights (~90MB), which is NOT "
                f"available in every environment (see this class's docstring): {e}"
            ) from e

    def _normalize(self, vecs: list) -> np.ndarray:
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = list(self._model.embed(texts))
        return self._normalize(vecs).tolist()

    def embed_query(self, text: str) -> list[float]:
        vecs = list(self._model.embed([text]))
        return self._normalize(vecs)[0].tolist()


class OpenAIEmbeddingsBackend(Embeddings):
    """Wrapper around ``langchain_openai.OpenAIEmbeddings``.

    Kept for portability per the original spec, but NOT exercised by this
    repo's own runs: api.openai.com was unreachable from the build sandbox.
    Requires ``pip install langchain-openai`` and ``OPENAI_API_KEY``.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from e
        self._impl = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)


def get_embeddings(backend: str | None = None) -> Embeddings:
    """Factory selecting the embedding backend from arg or EMBEDDING_BACKEND env var."""
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "spacy")).lower()
    if backend == "spacy":
        return SpacyVectorEmbeddings(os.getenv("SPACY_MODEL", DEFAULT_SPACY_MODEL))
    if backend in ("minilm", "sentence-transformer", "sentence-transformers"):
        return SentenceTransformerEmbeddings()
    if backend == "openai":
        return OpenAIEmbeddingsBackend(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} (expected 'spacy', 'minilm', or 'openai')")
