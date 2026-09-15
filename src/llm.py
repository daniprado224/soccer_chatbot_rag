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

LLM_BACKEND = os.getenv("LLM_BACKEND", "claude").lower()

# Model name defaults are backend-specific -- e.g. a Claude model name
# while LLM_BACKEND=gemini would just 404, and vice versa.
if LLM_BACKEND == "gemini":
    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    # Flash is already cheap/fast enough to use for grading too, unlike
    # the Claude default where grading gets its own smaller model.
    GRADER_MODEL = os.getenv("GEMINI_GRADER_MODEL", DEFAULT_MODEL)
else:
    DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    GRADER_MODEL = os.getenv("CLAUDE_GRADER_MODEL", "claude-haiku-4-5-20251001")


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def _call_claude(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic

    # Org-scoped (not workspace-scoped) API keys require an explicit
    # anthropic-workspace-id header on every request.
    workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(default_headers=headers)  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_gemini(system: str, user: str, model: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # free tier via aistudio.google.com
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
    )
    return resp.text or ""


def call_llm(system: str, user: str, model: str, max_tokens: int = 1024) -> str:
    if LLM_BACKEND == "gemini":
        return _call_gemini(system, user, model, max_tokens)
    return _call_claude(system, user, model, max_tokens)
