"""Unit tests for src/embeddings.py's backend factory and the new
SentenceTransformerEmbeddings (minilm) backend.

fastembed itself needs network access to huggingface.co on first use to
download model weights, which is not guaranteed to be available wherever
tests run (it wasn't, in this repo's own dev sandbox -- see the class's
docstring). So these tests mock fastembed's TextEmbedding entirely: they
verify this repo's own code (normalization, the Embeddings interface,
factory wiring, error messages), not fastembed's model quality.
"""
import sys
import types

import numpy as np
import pytest

import src.embeddings as embeddings_mod


class _FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: returns fixed, deliberately
    non-unit-length vectors so normalization can be verified."""

    last_init_kwargs = None

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        _FakeTextEmbedding.last_init_kwargs = kwargs

    def embed(self, texts):
        return [np.array([3.0, 4.0], dtype=np.float32) for _ in texts]  # norm = 5


@pytest.fixture
def fake_fastembed(monkeypatch):
    fake_module = types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    yield
    _FakeTextEmbedding.last_init_kwargs = None


def test_sentence_transformer_embeddings_normalizes_vectors(fake_fastembed):
    backend = embeddings_mod.SentenceTransformerEmbeddings()

    vec = backend.embed_query("what is an offside?")

    assert vec == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_sentence_transformer_embeddings_embed_documents_normalizes_each_row(fake_fastembed):
    backend = embeddings_mod.SentenceTransformerEmbeddings()

    vecs = backend.embed_documents(["a", "b", "c"])

    assert len(vecs) == 3
    for v in vecs:
        assert np.linalg.norm(v) == pytest.approx(1.0)


def test_sentence_transformer_embeddings_uses_tmp_cache_dir_on_vercel(fake_fastembed, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("FASTEMBED_CACHE_DIR", raising=False)

    embeddings_mod.SentenceTransformerEmbeddings()

    assert _FakeTextEmbedding.last_init_kwargs == {"cache_dir": "/tmp/fastembed_cache"}


def test_sentence_transformer_embeddings_raises_a_clear_error_when_fastembed_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # simulates ImportError on `from fastembed import ...`

    with pytest.raises(RuntimeError, match="fastembed is not installed"):
        embeddings_mod.SentenceTransformerEmbeddings()


@pytest.mark.parametrize("key", ["minilm", "sentence-transformer", "sentence-transformers"])
def test_get_embeddings_factory_recognizes_minilm_aliases(fake_fastembed, key):
    backend = embeddings_mod.get_embeddings(key)

    assert isinstance(backend, embeddings_mod.SentenceTransformerEmbeddings)


def test_get_embeddings_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_BACKEND"):
        embeddings_mod.get_embeddings("made-up-backend")
