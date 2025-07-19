import pandas as pd
import pytest
from datetime import timezone

from src import ingest


@pytest.mark.asyncio
async def test_yahoo_close_column(monkeypatch):
    df_raw = pd.DataFrame({"Close": [7, 8]}, index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    df_raw.index.name = "Date"

    def fake_download(*args, **kwargs):
        return df_raw

    monkeypatch.setattr(ingest.yf, "download", fake_download)

    df = await ingest._fetch_yahoo_gold()
    assert list(df.columns) == ["gold_price"]
    assert df.index.tz == timezone.utc
    assert df.iloc[0]["gold_price"] == 8
