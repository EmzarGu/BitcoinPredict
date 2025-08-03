import pandas as pd
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
        return pd.DataFrame({"close_usd": [10], "volume": [1]}, index=[pd.Timestamp(now)])

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        df = pd.DataFrame({"realised_price": [1], "nupl": [0]}, index=[week_start])
        return df

    async def fake_fetch_yahoo_series(ticker, column_name):
        df = pd.DataFrame({column_name: [5]}, index=[prev_week])
        return df

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_yahoo_series", fake_fetch_yahoo_series)

    df = await ingest.ingest_weekly()
    assert df["gold_price"].isna().sum() == 0


@pytest.mark.asyncio
async def test_week_start_present(monkeypatch):
    async def fake_fetch_coingecko(client, *args, **kwargs):
        week = pd.Timestamp("2024-01-01", tz="UTC")
        return pd.DataFrame({"close_usd": [10], "volume": [1]}, index=[week])

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        week = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame({"realised_price": [1], "nupl": [1]}, index=[week])
        return df

    async def fake_fetch_yahoo_series(ticker, column_name):
        week = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame({column_name: [1]}, index=[week])
        return df

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_yahoo_series", fake_fetch_yahoo_series)

    df = await ingest.ingest_weekly()
    assert df["week_start"].notna().all()
