"""Precompute embeddings for the deployed web app's in-memory vector store.

Run once (locally, whenever data/processed/chunks.json changes) after
`python -m src.ingestion`. Reads the already-ingested chunks and writes a
small .npy matrix per embedding backend alongside them; the web app loads
whichever ones it needs at cold start instead of talking to Chroma (see
src/memory_vectorstore.py for why).

    python -m scripts.build_embeddings_artifact               # builds spacy + minilm
    python -m scripts.build_embeddings_artifact --backend spacy
    python -m scripts.build_embeddings_artifact --backend minilm

The `minilm` backend needs network access to huggingface.co on first run
(to download the ONNX model weights) -- see
src/embeddings.py's SentenceTransformerEmbeddings docstring if it fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.json"

ALL_BACKENDS = ["spacy", "minilm"]


def build_one(backend: str, texts: list[str]) -> None:
    from src.embeddings import get_embeddings

    out_path = REPO_ROOT / "data" / "processed" / f"embeddings_{backend}.npy"
    print(f"[build_embeddings_artifact] embedding {len(texts)} chunks with backend={backend!r} ...")
    embeddings = get_embeddings(backend)
    vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)
    np.save(out_path, vectors)
    print(f"[build_embeddings_artifact] wrote {vectors.shape} to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=ALL_BACKENDS, default=None, help="Default: build all backends")
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text())
    texts = [c["text"] for c in chunks]
    backends = [args.backend] if args.backend else ALL_BACKENDS
    for backend in backends:
        build_one(backend, texts)


if __name__ == "__main__":
    main()
