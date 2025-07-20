import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from src import ingest


@pytest.mark.asyncio
async def test_forward_fill(monkeypatch):
    now = datetime(2024, 1, 5, tzinfo=timezone.utc)  # Friday
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    prev_week = week_start - timedelta(days=7)

    async def fake_fetch_coingecko(client, *args, **kwargs):
        ts = int(now.timestamp() * 1000)
        return {"prices": [[ts, 10]], "total_volumes": [[ts, 1]]}

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        df = pd.DataFrame({"realised_price": [1], "nupl": [0]}, index=[week_start])
        return df

    async def fake_fetch_fred_series(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        df = pd.DataFrame({col: [5]}, index=[prev_week])
        return df

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)

    df = await ingest.ingest_weekly()
    assert df["gold_price"].isna().sum() == 0


@pytest.mark.asyncio
async def test_week_start_present(monkeypatch):
    async def fake_fetch_coingecko(client, *args, **kwargs):
        ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        return {"prices": [[ts, 10]], "total_volumes": [[ts, 1]]}

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        week = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame({"realised_price": [1], "nupl": [1]}, index=[week])
        return df

    async def fake_fetch_fred_series(client, series_id):
        week = pd.Timestamp("2024-01-01", tz="UTC")
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        df = pd.DataFrame({col: [1]}, index=[week])
        return df

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)

    df = await ingest.ingest_weekly()
    assert df["week_start"].notna().all()
