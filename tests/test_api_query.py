"""Smoke tests for the actual deployed entrypoint files (api/query.py and
api/minilm/query.py), as opposed to tests/test_webapp.py's thorough
mocked-graph tests against the shared src/webapp.py factory.

These import the REAL modules (no mocking): api/query.py builds a real
spaCy graph against this repo's actual data/processed/ artifacts (same
as always ran in this sandbox), and api/minilm/query.py builds against
whatever's actually available -- which in this sandbox is nothing
(data/processed/embeddings_minilm.npy doesn't exist here; see
src/embeddings.py's SentenceTransformerEmbeddings docstring for why), so
it should come up cleanly with its backend marked unavailable rather
than crash at import time. The point of these tests is exactly that:
catching import-time/route-wiring regressions in the split into two
functions, not re-testing logic tests/test_webapp.py already covers.
"""
import importlib


def test_spacy_entrypoint_serves_the_unprefixed_api_routes():
    # Deliberately doesn't POST a real question through /api/query here --
    # that would fire a real Gemini API call (needs a key, costs money,
    # and would make this test flaky/network-dependent). Route wiring is
    # exactly what this test is for; query() behavior is already covered
    # thoroughly, with the LLM mocked, in tests/test_webapp.py.
    import api.query as api_query

    importlib.reload(api_query)  # in case an earlier test left module state
    client = api_query.app.test_client()

    assert client.get("/api/status").status_code == 200
    assert any(rule.rule == "/api/query" for rule in api_query.app.url_map.iter_rules())


def test_minilm_entrypoint_serves_the_prefixed_routes_and_503s_cleanly():
    import api.minilm.query as api_minilm_query

    importlib.reload(api_minilm_query)
    client = api_minilm_query.app.test_client()

    assert client.get("/api/minilm/status").status_code == 200
    resp = client.post("/api/minilm/query", json={"question": "what is an offside?"})
    # This sandbox never built data/processed/embeddings_minilm.npy (needs
    # real network access to huggingface.co -- see README), so this must
    # 503 with a clear message, not 500/crash.
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "embedding_model_unavailable"
