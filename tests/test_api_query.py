"""Tests for api/query.py's error-vs-genuine-refusal distinction.

Monkeypatches the already-built _graph object's query() method directly,
rather than mocking the LLM/vectorstore underneath it -- this is testing
the Flask route's own logic (does it correctly read grading_error/
generation_error off the graph's result), not the graph itself.
"""
import api.query as api_query


def _fake_state(**overrides):
    base = {
        "answer": "I don't know.",
        "cited_laws": [],
        "answerable": False,
        "retry_count": 1,
        "retrieved": [],
        "grading_error": False,
        "generation_error": False,
        "quota_info": None,
    }
    base.update(overrides)
    return base


def test_infra_error_returns_503_with_gemini_down_message(monkeypatch):
    monkeypatch.setattr(api_query, "_graph", type("G", (), {"query": staticmethod(lambda q: _fake_state(grading_error=True))})())
    monkeypatch.setattr(api_query.quota_store, "get_status", lambda: {"available": True})

    client = api_query.app.test_client()
    resp = client.post("/api/query", json={"question": "how many players are on each team?"})

    assert resp.status_code == 503
    data = resp.get_json()
    assert data["error"] == "temporarily_unavailable"
    assert "Gemini" in data["message"]


def test_genuine_no_match_is_not_treated_as_an_infra_error(monkeypatch):
    monkeypatch.setattr(api_query, "_graph", type("G", (), {"query": staticmethod(lambda q: _fake_state())})())
    monkeypatch.setattr(api_query.quota_store, "get_status", lambda: {"available": True})
    monkeypatch.setattr(api_query.quota_store, "record_question", lambda: None)
    monkeypatch.setattr(api_query, "suggest_better_phrasing", lambda *a, **k: None)

    client = api_query.app.test_client()
    resp = client.post("/api/query", json={"question": "what is an offside?"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answerable"] is False
    assert data["answer"] == "I don't know."


def test_generation_error_also_treated_as_infra_error(monkeypatch):
    monkeypatch.setattr(
        api_query, "_graph", type("G", (), {"query": staticmethod(lambda q: _fake_state(generation_error=True))})()
    )
    monkeypatch.setattr(api_query.quota_store, "get_status", lambda: {"available": True})

    client = api_query.app.test_client()
    resp = client.post("/api/query", json={"question": "q"})

    assert resp.status_code == 503
    assert resp.get_json()["error"] == "temporarily_unavailable"
