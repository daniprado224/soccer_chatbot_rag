"""Pluggable LLM backends for the grading/rewrite/answer nodes in graph.py.

Defaults to Claude, per the original spec. Gemini is wired in as a
free-tier alternative: Anthropic Console API access requires purchased
credits, and this repo's own eval needed to actually run without that.
`generativelanguage.googleapis.com` is also one of the only LLM-provider
hosts reachable from the sandbox this was built in -- Groq, Together,
OpenRouter, and Cohere were all network-blocked the same way OpenAI was.

Swap backends via the LLM_BACKEND env var ("claude" | "gemini"); nothing
in graph.py needs to change either way, since both paths return a plain
string. The tradeoff, worth being explicit about: this is a genuine model
swap, not just an infra detail -- eval numbers produced with
LLM_BACKEND=gemini measure Gemini's grading/generation quality, not
Claude's.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

# .strip() matters here, not just cosmetically: a trailing newline/space in
# an env var (easy to introduce via `vercel env add`'s interactive prompt,
# or copy-pasting) makes "gemini\n" != "gemini", silently falling through
# to the Claude default with no error -- which is exactly what happened
# testing the real Vercel deployment: LLM_BACKEND was set and visible in
# `vercel env ls`, but every request still hit Claude's "no credentials"
# error because the equality check below was failing silently.
LLM_BACKEND = os.getenv("LLM_BACKEND", "claude").strip().lower()

# Model name defaults are backend-specific -- e.g. a Claude model name
# while LLM_BACKEND=gemini would just 404, and vice versa.
if LLM_BACKEND == "gemini":
    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
    # Flash is already cheap/fast enough to use for grading too, unlike
    # the Claude default where grading gets its own smaller model.
    GRADER_MODEL = os.getenv("GEMINI_GRADER_MODEL", DEFAULT_MODEL).strip()
else:
    DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5").strip()
    GRADER_MODEL = os.getenv("CLAUDE_GRADER_MODEL", "claude-haiku-4-5-20251001").strip()


class GeminiQuotaExceeded(Exception):
    """A Gemini free-tier quota was actually exhausted (not a transient blip).

    Carries the API's own suggested retry delay and which quota dimension
    tripped (per-minute vs per-day, from the 429's quotaId) so callers --
    notably the deployed web app's shared quota tracker -- can show an
    accurate "try again around HH:MM" instead of guessing at limits that
    were only ever empirically observed, not officially documented for
    this exact model.
    """

    def __init__(self, retry_after_seconds: float, scope: str):
        self.retry_after_seconds = retry_after_seconds
        self.scope = scope  # "minute" | "day" | "unknown"
        super().__init__(f"Gemini quota exceeded ({scope}), retry after {retry_after_seconds:.0f}s")


def _parse_quota_exhaustion(message: str) -> tuple[float, str]:
    delay_match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", message)
    retry_after = float(delay_match.group(1)) if delay_match else 60.0
    if "PerDay" in message:
        scope = "day"
    elif "PerMinute" in message:
        scope = "minute"
    else:
        scope = "unknown"
    return retry_after, scope


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def _call_claude(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic

    # Explicit .strip()'d reads throughout this module, not just for
    # LLM_BACKEND above -- an API key with trailing whitespace from the
    # same class of env-var-entry mistake would otherwise fail auth with a
    # confusing error instead of just working.
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
    workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip() or None
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(api_key=api_key, default_headers=headers)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# The free tier enforces a per-minute request cap (observed: 15 req/min
# for gemini-3.1-flash-lite) in addition to whatever daily cap it also
# has. eval.py fires calls back-to-back with no natural spacing, so
# without this a multi-question run reliably blows through it after a
# handful of questions. Paced to ~10/min (6s apart) to leave margin rather
# than ride the exact limit, plus one retry on a 429 since a burst from
# some other process against the same key/project could still trip it.
_GEMINI_MIN_INTERVAL_SECONDS = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "6"))
_gemini_call_lock = threading.Lock()
_gemini_last_call_time = 0.0


def _throttle_gemini() -> None:
    global _gemini_last_call_time
    with _gemini_call_lock:
        wait = _gemini_last_call_time + _GEMINI_MIN_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _gemini_last_call_time = time.monotonic()


def _call_gemini(system: str, user: str, model: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())  # free tier via aistudio.google.com
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        # Flash models "think" (hidden reasoning tokens) by default, which
        # can consume the entire max_output_tokens budget before any
        # visible text is produced -- these are short structured
        # classification/JSON tasks with no need for chain-of-thought.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    for attempt in range(2):
        _throttle_gemini()
        try:
            resp = client.models.generate_content(model=model, contents=user, config=config)
            return resp.text or ""
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                if attempt == 0:
                    time.sleep(15)  # one retry past a transient per-minute cap; a daily cap will still raise
                    continue
                retry_after, scope = _parse_quota_exhaustion(str(e))
                raise GeminiQuotaExceeded(retry_after, scope) from e
            raise


def call_llm(system: str, user: str, model: str, max_tokens: int = 1024) -> str:
    if LLM_BACKEND == "gemini":
        return _call_gemini(system, user, model, max_tokens)
    return _call_claude(system, user, model, max_tokens)
