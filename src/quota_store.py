"""Shared quota/usage tracking for the deployed web app, via Upstash Redis.

Why this needs to exist at all: every visitor to the deployed app hits the
SAME Gemini API key. The free tier's quota is a resource shared across all
of them, not per-visitor -- so a per-browser (localStorage) countdown would
be actively misleading (it can't see what other visitors' requests already
used). Tracking has to live server-side, in something all function
invocations can read and write.

Why reactive, not predictive: Google doesn't expose a "quota remaining"
endpoint to free-tier callers. The only two limits this project has
directly confirmed are both from real 429 responses hit during
development (see README) -- 20/day for gemini-3.6-flash, 15/min for
gemini-3.1-flash-lite -- and those are specific to models this project
happened to try, not a documented guarantee for whatever model is
configured. So instead of asserting a number and counting down to it
(which would be a guess dressed up as a fact), this records the ACTUAL
moment a real 429 happens, with the API's own suggested retry delay, and
reports "temporarily unavailable, resets around HH:MM" only once that has
actually occurred -- otherwise it just reports "available" plus how many
questions have been served today, without pretending to know how close to
the ceiling that is.

Uses Upstash Redis's REST API directly (plain HTTPS + requests) rather
than a Redis client library, since that's what Vercel's native Upstash
integration exposes via UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN
env vars, and it works identically from any Python runtime without a
persistent connection.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from urllib.parse import quote

import requests

QUESTIONS_KEY_PREFIX = "soccer_rag:questions"
EXHAUSTED_KEY = "soccer_rag:quota_exhausted"
_TIMEOUT = 3  # seconds -- a quota-store hiccup should never hang a user's question


def _base_url() -> str | None:
    return os.getenv("UPSTASH_REDIS_REST_URL")


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.getenv('UPSTASH_REDIS_REST_TOKEN', '')}"}


def _today_key() -> str:
    return f"{QUESTIONS_KEY_PREFIX}:{dt.date.today().isoformat()}"


def _configured() -> bool:
    return bool(_base_url() and os.getenv("UPSTASH_REDIS_REST_TOKEN"))


def record_question() -> None:
    """Increment today's served-question counter. Best-effort: a tracker
    outage must never block an actual answer from being returned."""
    if not _configured():
        return
    try:
        key = _today_key()
        requests.get(f"{_base_url()}/incr/{key}", headers=_headers(), timeout=_TIMEOUT)
        requests.get(f"{_base_url()}/expire/{key}/172800", headers=_headers(), timeout=_TIMEOUT)  # 2 days
    except requests.RequestException:
        pass


def mark_exhausted(retry_after_seconds: float, scope: str) -> None:
    """Record a real quota exhaustion so /api/status can report it accurately."""
    if not _configured():
        return
    try:
        until = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=retry_after_seconds)).isoformat()
        payload = quote(json.dumps({"until": until, "scope": scope}))
        ttl = max(30, min(int(retry_after_seconds) + 30, 86400))  # small buffer, capped at 1 day
        requests.get(f"{_base_url()}/set/{EXHAUSTED_KEY}/{payload}/EX/{ttl}", headers=_headers(), timeout=_TIMEOUT)
    except requests.RequestException:
        pass


def get_status() -> dict:
    """Returns {questions_today, available, reset_at, scope, tracker_configured}."""
    if not _configured():
        return {
            "questions_today": None,
            "available": True,
            "reset_at": None,
            "scope": None,
            "tracker_configured": False,
        }

    questions_today = 0
    reset_at = None
    scope = None
    try:
        resp = requests.get(f"{_base_url()}/get/{_today_key()}", headers=_headers(), timeout=_TIMEOUT)
        result = resp.json().get("result")
        questions_today = int(result) if result else 0

        resp = requests.get(f"{_base_url()}/get/{EXHAUSTED_KEY}", headers=_headers(), timeout=_TIMEOUT)
        result = resp.json().get("result")
        if result:
            data = json.loads(result)
            reset_at = data.get("until")
            scope = data.get("scope")
    except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError):
        pass  # a status-check hiccup should degrade to "available", not crash the page

    available = reset_at is None or dt.datetime.fromisoformat(reset_at) <= dt.datetime.now(dt.timezone.utc)
    return {
        "questions_today": questions_today,
        "available": available,
        "reset_at": reset_at if not available else None,
        "scope": scope if not available else None,
        "tracker_configured": True,
    }
