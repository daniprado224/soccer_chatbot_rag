"""Vercel serverless function: the MiniLM-embedding half of the deployed
web demo (GET /api/minilm/status, POST /api/minilm/query).

Deliberately a separate function from ../query.py's spaCy backend --
see src/webapp.py's docstring for why (a single function bundling both
backends exceeded Vercel's 500MB function size limit). This file's own
requirements.txt (api/minilm/requirements.txt) pulls in fastembed +
onnxruntime but NOT spacy/en_core_web_md, so this bundle never pays for
the other backend's weight either.

All actual route logic lives in src/webapp.py so both functions share
identical behavior. If data/processed/embeddings_minilm.npy was never
built for this deployment, requests here cleanly 503 rather than crash
-- see build_app()'s docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.webapp import build_app  # noqa: E402

app = build_app("minilm", route_prefix="/api/minilm")
