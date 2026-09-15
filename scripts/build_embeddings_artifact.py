"""Precompute embeddings for the deployed web app's in-memory vector store.

Run once (locally, whenever data/processed/chunks.json changes) after
`python -m src.ingestion`. Reads the already-ingested chunks and writes a
small .npy matrix alongside them; the web app loads both at cold start
instead of talking to Chroma (see src/memory_vectorstore.py for why).

    python -m scripts.build_embeddings_artifact
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.json"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.npy"


def main() -> None:
    from src.embeddings import get_embeddings

    chunks = json.loads(CHUNKS_PATH.read_text())
    texts = [c["text"] for c in chunks]
    print(f"[build_embeddings_artifact] embedding {len(texts)} chunks from {CHUNKS_PATH} ...")

    embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)

    np.save(EMBEDDINGS_PATH, vectors)
    print(f"[build_embeddings_artifact] wrote {vectors.shape} to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
