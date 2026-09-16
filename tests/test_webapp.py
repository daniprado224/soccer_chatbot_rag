"""Tests for src/webapp.py's Flask app factory: the shared route logic
behind both deployed functions (api/query.py = spacy, api/minilm/query.py
= minilm). See that module's docstring for why it's split into two
functions in the first place (a Vercel function size limit).

Monkeypatches src.webapp._build_graph to hand back a fake graph object
whose query() returns a canned state, rather than mocking the LLM/vector
store underneath it -- this is testing the Flask route's own logic (error
classification, embedding_model validation, route_prefix wiring), not the
graph itself.
"""
import src.webapp as webapp


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


def _fake_graph(state, vectorstore=None):
    return type("G", (), {"query": staticmethod(lambda q: state), "vectorstore": vectorstore})()


def _build_test_app(monkeypatch, embedding_model, state, route_prefix="/api"):
    monkeypatch.setattr(webapp, "_build_graph", lambda model: _fake_graph(state))
    monkeypatch.setattr(webapp.quota_store, "get_status", lambda: {"available": True})
    monkeypatch.setattr(webapp.quota_store, "record_question", lambda: None)
    monkeypatch.setattr(webapp, "suggest_better_phrasing", lambda *a, **k: None)
    app = webapp.build_app(embedding_model, route_prefix=route_prefix)
    return app.test_client()


def test_infra_error_returns_503_with_gemini_down_message(monkeypatch):
    client = _build_test_app(monkeypatch, "spacy", _fake_state(grading_error=True))

    resp = client.post("/api/query", json={"question": "how many players are on each team?"})

    assert resp.status_code == 503
    data = resp.get_json()
    assert data["error"] == "temporarily_unavailable"
    assert "Gemini" in data["message"]


def test_genuine_no_match_is_not_treated_as_an_infra_error(monkeypatch):
    client = _build_test_app(monkeypatch, "spacy", _fake_state())

    resp = client.post("/api/query", json={"question": "what is an offside?"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answerable"] is False
    assert data["answer"] == "I don't know."
    assert data["embedding_model"] == "spacy"


def test_generation_error_also_treated_as_infra_error(monkeypatch):
    client = _build_test_app(monkeypatch, "spacy", _fake_state(generation_error=True))

    resp = client.post("/api/query", json={"question": "q"})

    assert resp.status_code == 503
    assert resp.get_json()["error"] == "temporarily_unavailable"


def test_app_reports_its_own_fixed_embedding_model(monkeypatch):
    client = _build_test_app(
        monkeypatch, "minilm", _fake_state(answer="hi", answerable=True), route_prefix="/api/minilm"
    )

    resp = client.post("/api/minilm/query", json={"question": "q"})

    assert resp.status_code == 200
    assert resp.get_json()["embedding_model"] == "minilm"


def test_query_rejects_a_mismatched_embedding_model(monkeypatch):
    client = _build_test_app(monkeypatch, "spacy", _fake_state())

    resp = client.post("/api/query", json={"question": "q", "embedding_model": "minilm"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "wrong_endpoint"


def test_build_app_rejects_unknown_embedding_model():
    try:
        webapp.build_app("made-up-model")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_route_prefix_controls_which_paths_are_registered(monkeypatch):
    monkeypatch.setattr(webapp, "_build_graph", lambda model: _fake_graph(_fake_state()))

    app = webapp.build_app("minilm", route_prefix="/api/minilm")
    client = app.test_client()

    assert client.get("/api/minilm/status").status_code == 200
    assert client.get("/api/status").status_code == 404


def test_query_returns_503_when_embedding_model_not_built_on_this_deployment(monkeypatch):
    monkeypatch.setattr(
        webapp,
        "_build_graph",
        lambda model: (_ for _ in ()).throw(RuntimeError(f"'{model}' embedding backend isn't set up")),
    )
    monkeypatch.setattr(webapp.quota_store, "get_status", lambda: {"available": True})

    app = webapp.build_app("minilm", route_prefix="/api/minilm")
    client = app.test_client()

    resp = client.post("/api/minilm/query", json={"question": "q"})

    assert resp.status_code == 503
    assert resp.get_json()["error"] == "embedding_model_unavailable"


def test_status_route_works_even_when_this_functions_own_backend_failed_to_load(monkeypatch):
    monkeypatch.setattr(
        webapp, "_build_graph", lambda model: (_ for _ in ()).throw(RuntimeError("missing artifact"))
    )
    monkeypatch.setattr(webapp.quota_store, "get_status", lambda: {"available": True})

    app = webapp.build_app("minilm", route_prefix="/api/minilm")
    client = app.test_client()

    resp = client.get("/api/minilm/status")

    assert resp.status_code == 200
    assert "embedding_models" in resp.get_json()
