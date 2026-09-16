"""Vercel serverless function: the spaCy-embedding half of the deployed
web demo (GET /api/status, POST /api/query).

Deliberately a separate function from api/minilm/query.py's MiniLM
backend, purely for deployment bundle size -- see src/webapp.py's
docstring. This file only bundles spaCy + en_core_web_md, never
fastembed/onnxruntime, so it stays well under Vercel's 500MB function
limit regardless of what the minilm function needs.

All actual route logic lives in src/webapp.py so both functions share
identical behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.webapp import build_app  # noqa: E402

app = build_app("spacy", route_prefix="/api")
