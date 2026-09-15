"""Embedding backends behind a common LangChain-compatible interface.

Only ``spacy`` is actually exercised end-to-end by this repo's own ingestion
and eval runs: the dev sandbox this project was built in has network egress
blocked to both api.openai.com and huggingface.co, so OpenAI embeddings and
HuggingFace-hosted local models could not be downloaded or called from here.
``en_core_web_md`` ships as a plain pip wheel from a GitHub release asset,
which was reachable, so it is the default.

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
    if backend == "openai":
        return OpenAIEmbeddingsBackend(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} (expected 'spacy' or 'openai')")
