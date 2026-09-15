"""Unit tests for the quota tracker's logic, with requests.get mocked --
no real Upstash instance or network access needed."""
import json

import pytest

import src.quota_store as quota_store


@pytest.fixture(autouse=True)
def upstash_configured(monkeypatch):
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake-upstash.example.com")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")


class FakeResponse:
    def __init__(self, result):
        self._result = result

    def json(self):
        return {"result": self._result}


def test_get_status_reports_unconfigured_tracker_as_always_available(monkeypatch):
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    status = quota_store.get_status()
    assert status == {
        "questions_today": None,
        "available": True,
        "reset_at": None,
        "scope": None,
        "tracker_configured": False,
    }


def test_get_status_available_when_no_exhaustion_recorded(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if url.endswith(quota_store._today_key()):
            return FakeResponse("7")
        return FakeResponse(None)

    monkeypatch.setattr(quota_store.requests, "get", fake_get)

    status = quota_store.get_status()
    assert status["questions_today"] == 7
    assert status["available"] is True
    assert status["reset_at"] is None


def test_get_status_unavailable_when_exhaustion_is_in_the_future(monkeypatch):
    import datetime as dt

    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)).isoformat()

    def fake_get(url, headers=None, timeout=None):
        if url.endswith(quota_store._today_key()):
            return FakeResponse("3")
        return FakeResponse(json.dumps({"until": future, "scope": "minute"}))

    monkeypatch.setattr(quota_store.requests, "get", fake_get)

    status = quota_store.get_status()
    assert status["available"] is False
    assert status["reset_at"] == future
    assert status["scope"] == "minute"


def test_get_status_available_again_once_exhaustion_timestamp_has_passed(monkeypatch):
    import datetime as dt

    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()

    def fake_get(url, headers=None, timeout=None):
        if url.endswith(quota_store._today_key()):
            return FakeResponse("10")
        return FakeResponse(json.dumps({"until": past, "scope": "day"}))

    monkeypatch.setattr(quota_store.requests, "get", fake_get)

    status = quota_store.get_status()
    assert status["available"] is True
    assert status["reset_at"] is None


def test_record_question_and_mark_exhausted_never_raise_on_network_error(monkeypatch):
    import requests as real_requests

    def raising_get(*args, **kwargs):
        raise real_requests.RequestException("network is down")

    monkeypatch.setattr(quota_store.requests, "get", raising_get)

    quota_store.record_question()  # must not raise
    quota_store.mark_exhausted(30.0, "minute")  # must not raise
