import pandas as pd
import pytest
import httpx
from datetime import datetime, timezone, timedelta

from src import ingest


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def get(self, *args, **kwargs):
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp


@pytest.mark.asyncio
async def test_historical_anchor_uses_yahoo(monkeypatch):
    """_fetch_coingecko should use Yahoo Finance when a range is requested."""
    week = datetime.now(timezone.utc) - timedelta(days=400)
    yahoo_calls = 0

    async def fake_yahoo(*args, **kwargs):
        nonlocal yahoo_calls
        yahoo_calls += 1
        return pd.DataFrame({"close_usd": [5], "volume": [pd.NA]}, index=[week])

    httpx_called = False

    class NoCallClient(FakeClient):
        async def get(self, *a, **k):
            nonlocal httpx_called
            httpx_called = True
            return await super().get(*a, **k)

    client = NoCallClient([FakeResp(200)])
    monkeypatch.setattr(ingest, "_fetch_yahoo_btc", fake_yahoo)

    df = await ingest._fetch_coingecko(client, start=week, end=week + timedelta(days=7))

    assert yahoo_calls == 1
    assert not httpx_called
    assert list(df.columns) == ["close_usd", "volume"]


@pytest.mark.asyncio
async def test_header_added(monkeypatch):
    week = datetime.now(timezone.utc)
    payload = {
        "prices": [[int(week.timestamp() * 1000), 10]],
        "total_volumes": [[int(week.timestamp() * 1000), 1]],
    }
    resp = FakeResp(200, payload)

    captured = {}

    class HeaderClient(FakeClient):
        async def get(self, url, params=None, headers=None, timeout=30):
            captured["headers"] = headers
            return await super().get(url, params=params, headers=headers, timeout=timeout)

    client = HeaderClient([resp])
    monkeypatch.setenv("COINGECKO_API_KEY", "abc")
    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda *a, **k: client)

    df = await ingest._fetch_coingecko(client)
    assert "x-cg-demo-api-key" in captured.get("headers", {})
    assert not df.empty
