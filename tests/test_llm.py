from src.llm import _parse_quota_exhaustion

REAL_DAY_QUOTA_MESSAGE = (
    "429 RESOURCE_EXHAUSTED. {'error': {..., 'details': [..., "
    "{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', ...}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '57s'}]}}"
)

REAL_MINUTE_QUOTA_MESSAGE = (
    "429 RESOURCE_EXHAUSTED. {'error': {..., 'details': [..., "
    "{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', ...}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '35s'}]}}"
)


def test_parses_real_daily_quota_message():
    retry_after, scope = _parse_quota_exhaustion(REAL_DAY_QUOTA_MESSAGE)
    assert retry_after == 57.0
    assert scope == "day"


def test_parses_real_minute_quota_message():
    retry_after, scope = _parse_quota_exhaustion(REAL_MINUTE_QUOTA_MESSAGE)
    assert retry_after == 35.0
    assert scope == "minute"


def test_falls_back_to_default_delay_and_unknown_scope_on_unparseable_message():
    retry_after, scope = _parse_quota_exhaustion("429 RESOURCE_EXHAUSTED. something unexpected")
    assert retry_after == 60.0
    assert scope == "unknown"
