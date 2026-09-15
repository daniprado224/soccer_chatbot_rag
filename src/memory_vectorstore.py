"""A tiny in-memory vector store, used only by the deployed web app.

Chroma (used everywhere else in this repo) persists to local disk, which
doesn't work on Vercel's serverless functions -- their filesystem is
ephemeral and not shared across invocations or concurrent instances. With
only 144 chunks in the real corpus (~172KB of float32 vectors), there's no
real need for an actual vector database in production: load a precomputed
embedding matrix once per cold start and brute-force cosine similarity
over it on every request. That's the entire "database".

Exposes the same `similarity_search_with_score(query, k)` shape Chroma
does, so RetrievalGraph in graph.py needs zero changes to use this instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class InMemoryVectorStore:
    def __init__(self, chunks_path: str | Path, embeddings_path: str | Path, embeddings_backend: Embeddings):
        chunks = json.loads(Path(chunks_path).read_text())
        self.docs = [
            Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"})
            for c in chunks
        ]
        # Rows are pre-normalized (unit length) at build time -- see
        # scripts/build_embeddings_artifact.py -- so a plain dot product
        # against a normalized query vector IS cosine similarity, no need
        # to re-normalize per query.
        self.matrix = np.load(embeddings_path)
        if len(self.docs) != self.matrix.shape[0]:
            raise ValueError(
                f"chunks/embeddings mismatch: {len(self.docs)} chunks vs {self.matrix.shape[0]} vectors"
            )
        self.embeddings_backend = embeddings_backend

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        qvec = np.array(self.embeddings_backend.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm
        sims = self.matrix @ qvec  # cosine similarity per row, higher = more similar
        top_k = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in top_k]
