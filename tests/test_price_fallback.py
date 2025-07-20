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
async def test_fallback_to_yahoo_on_401(monkeypatch):
    week = datetime.now(timezone.utc) - timedelta(days=400)
    payload = {"prices": [[int(week.timestamp() * 1000), 10]], "total_volumes": [[int(week.timestamp() * 1000), 1]]}
    client = FakeClient([FakeResp(401), FakeResp(401), FakeResp(401)])

    async def fake_yahoo(*args, **kwargs):
        return pd.DataFrame({"close_usd": [5], "volume": [pd.NA]}, index=[week])

    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setattr(ingest, "_fetch_yahoo_btc", fake_yahoo)

    df = await ingest._fetch_coingecko(client, start=week, end=week + timedelta(days=7))
    assert not df.empty
    assert pd.isna(df.iloc[0]["volume"])


@pytest.mark.asyncio
async def test_header_added(monkeypatch):
    week = datetime.now(timezone.utc)
    payload = {"prices": [[int(week.timestamp() * 1000), 10]], "total_volumes": [[int(week.timestamp() * 1000), 1]]}
    resp = FakeResp(200, payload)

    captured = {}

    class HeaderClient(FakeClient):
        async def get(self, url, params=None, headers=None, timeout=30):
            captured["headers"] = headers
            return await super().get(url, params=params, headers=headers, timeout=timeout)

    client = HeaderClient([resp])
    monkeypatch.setenv("COINGECKO_API_KEY", "abc")
    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda *a, **k: client)

    df = await ingest._fetch_coingecko(client, start=week, end=week + timedelta(days=7))
    assert "x-cg-demo-api-key" in captured.get("headers", {})
    assert not df.empty
