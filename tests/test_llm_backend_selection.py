"""Regression test for a real production bug: LLM_BACKEND set to "gemini"
via `vercel env add`'s interactive prompt still resolved to the Claude
default at runtime, because a trailing newline in the stored value made
"gemini\n" != "gemini". src/llm.py reads LLM_BACKEND at import time, so
this reimports the module fresh with a patched environment to verify the
fix, rather than testing the module-level constant directly.
"""
import importlib

import src.llm


def _reload_with_env(monkeypatch, **env):
    for key in ("LLM_BACKEND", "GEMINI_MODEL", "GEMINI_GRADER_MODEL", "CLAUDE_MODEL", "CLAUDE_GRADER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(src.llm)


def test_trailing_newline_in_llm_backend_still_selects_gemini(monkeypatch):
    _reload_with_env(monkeypatch, LLM_BACKEND="gemini\n")
    assert src.llm.LLM_BACKEND == "gemini"


def test_trailing_space_and_mixed_case_still_selects_gemini(monkeypatch):
    _reload_with_env(monkeypatch, LLM_BACKEND="  Gemini  ")
    assert src.llm.LLM_BACKEND == "gemini"


def test_model_name_env_vars_are_also_stripped(monkeypatch):
    _reload_with_env(monkeypatch, LLM_BACKEND="gemini", GEMINI_MODEL="gemini-3.1-flash-lite\n")
    assert src.llm.DEFAULT_MODEL == "gemini-3.1-flash-lite"


def teardown_module(module):
    # Leave the module in its default (env-free) state for any other test
    # file that imports it after this one runs.
    import os

    for key in ("LLM_BACKEND", "GEMINI_MODEL", "GEMINI_GRADER_MODEL", "CLAUDE_MODEL", "CLAUDE_GRADER_MODEL"):
        os.environ.pop(key, None)
    importlib.reload(src.llm)
