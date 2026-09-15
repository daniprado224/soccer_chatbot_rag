"""Regression test: a real 503 "high demand" response from Gemini was
being treated as a permanent grading failure with no retry, turning a
perfectly answerable question into a false "I don't know" purely because
one call landed on a transient server hiccup. Mocks the google-genai
client so no real network/API key is needed.
"""
import pytest
from google.genai.errors import ServerError

import src.llm as llm_mod


class FakeResponse:
    def __init__(self, text):
        self.text = text


def _server_error():
    return ServerError(code=503, response_json={"error": {"message": "high demand"}})


class FakeModels:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)

    def generate_content(self, model, contents, config):
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return FakeResponse(effect)


class FakeClient:
    def __init__(self, side_effects):
        self.models = FakeModels(side_effects)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm_mod, "_GEMINI_MIN_INTERVAL_SECONDS", 0.0)


def test_recovers_from_a_single_transient_503(monkeypatch):
    fake_client = FakeClient([_server_error(), "a real answer"])
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    result = llm_mod._call_gemini("system", "user", "gemini-3.1-flash-lite", 100)

    assert result == "a real answer"


def test_recovers_from_two_consecutive_transient_503s(monkeypatch):
    fake_client = FakeClient([_server_error(), _server_error(), "a real answer"])
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    result = llm_mod._call_gemini("system", "user", "gemini-3.1-flash-lite", 100)

    assert result == "a real answer"


def test_raises_after_exhausting_retries_on_persistent_503(monkeypatch):
    fake_client = FakeClient([_server_error(), _server_error(), _server_error()])
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    with pytest.raises(ServerError):
        llm_mod._call_gemini("system", "user", "gemini-3.1-flash-lite", 100)
