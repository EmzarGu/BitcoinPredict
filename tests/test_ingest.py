from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone

import pandas as pd
import pytest

import src.ingest as ingest


def test_coin_gecko_to_weekly():
    ts0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    ts1 = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)
    ts2 = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp() * 1000)
    data = {
        "prices": [[ts0, 10], [ts1, 11], [ts2, 12]],
        "total_volumes": [[ts0, 1], [ts1, 2], [ts2, 3]],
    }
    df = ingest._coingecko_to_weekly(data)
    assert df.iloc[0]["close_usd"] == 12
    assert df.iloc[0]["volume"] == 6


@pytest.mark.asyncio
async def test_schema_columns(monkeypatch):
    week_start = pd.Timestamp("2024-01-01", tz="UTC")

    async def fake_fetch_coingecko(client):
        return {
            "prices": [[int(week_start.timestamp() * 1000), 10]],
            "total_volumes": [[int(week_start.timestamp() * 1000), 1]],
        }

    async def fake_fetch_coinmetrics(client):
        df = pd.DataFrame({
            "realised_price": [1],
            "nupl": [1],
        }, index=[week_start])
        return df

    async def fake_fetch_fred_series(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        df = pd.DataFrame({col: [1]}, index=[week_start])
        return df

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)

    df = await ingest.ingest_weekly()
    assert list(df.columns) == ingest.SCHEMA_COLUMNS
    assert df.loc[0, "fed_liq"] == 1
    assert df.loc[0, "dxy"] == 1
    assert df.loc[0, "ust10"] == 1


@pytest.mark.asyncio
async def test_fetch_fred_series(monkeypatch):
    csv = "DATE,WALCL\n2024-01-01,10\n"

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    class FakeClient:
        async def get(self, url, timeout=30):
            return FakeResponse(csv)

    df = await ingest._fetch_fred_series(FakeClient(), "WALCL")
    assert list(df.columns) == ["fed_liq"]
    assert df.iloc[0]["fed_liq"] == 10
