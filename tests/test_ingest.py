from datetime import datetime, timezone
import logging

import pandas as pd
import pytest
import warnings

import httpx
import psycopg2
from src import ingest


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
    week_start = pd.Timestamp(datetime.now(tz=timezone.utc))

    async def fake_fetch_coingecko(client, *args, **kwargs):
        df = pd.DataFrame({"close_usd": [10], "volume": [1]}, index=[week_start])
        return df

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        df = pd.DataFrame(
            {
                "realised_price": [1],
                "nupl": [1],
            },
            index=[week_start],
        )
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
    assert df.loc[0, "ecb_liq"] == 1
    assert df.loc[0, "dxy"] == 1
    assert df.loc[0, "ust10"] == 1
    assert df.loc[0, "gold_price"] == 1
    assert df.loc[0, "spx_index"] == 1


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


@pytest.mark.asyncio
async def test_fetch_fred_series_with_api_key(monkeypatch):
    csv = "DATE,WALCL\n2024-01-01,10\n"

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self):
            self.urls = []

        async def get(self, url, timeout=30):
            self.urls.append(url)
            return FakeResponse(csv)

    client = FakeClient()
    monkeypatch.setattr(ingest, "FRED_API_KEY", "testkey")

    df = await ingest._fetch_fred_series(client, "WALCL")
    assert list(df.columns) == ["fed_liq"]
    assert df.iloc[0]["fed_liq"] == 10
    assert client.urls[0].endswith("&api_key=testkey")


@pytest.mark.asyncio
async def test_fetch_fred_series_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError("error", request=None, response=self)

    class FakeClient:
        async def get(self, url, timeout=30):
            return FakeResponse()

    df = await ingest._fetch_fred_series(FakeClient(), "WALCL")
    assert list(df.columns) == ["fed_liq"]
    assert df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz == timezone.utc


@pytest.mark.asyncio
async def test_fetch_fred_series_request_error(monkeypatch):
    class FakeClient:
        async def get(self, url, timeout=30):
            raise httpx.RequestError("boom", request=httpx.Request("GET", url))

    df = await ingest._fetch_fred_series(FakeClient(), "WALCL")
    assert list(df.columns) == ["fed_liq"]
    assert df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz == timezone.utc


@pytest.mark.asyncio
async def test_ingest_weekly_fred_failure(monkeypatch):
    week_start = pd.Timestamp(datetime.now(tz=timezone.utc))

    async def fake_fetch_coingecko(client, *args, **kwargs):
        return pd.DataFrame({"close_usd": [10], "volume": [1]}, index=[week_start])

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        df = pd.DataFrame({"realised_price": [1], "nupl": [0]}, index=[week_start])
        return df

    async def fake_fetch_fred_series(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        return pd.DataFrame(columns=[col], index=pd.DatetimeIndex([], tz="UTC"))

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)

    with pd.option_context("future.no_silent_downcasting", True):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            df = await ingest.ingest_weekly()
    assert len(w) == 0
    assert pd.isna(df.loc[0, "fed_liq"])


@pytest.mark.asyncio
async def test_fetch_yahoo_gold(monkeypatch):
    df_raw = pd.DataFrame(
        {"Adj Close": [4, 5]},
        index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
    )
    df_raw.index.name = "Date"

    def fake_download(*args, **kwargs):
        return df_raw

    monkeypatch.setattr(ingest.yf, "download", fake_download)

    df = await ingest._fetch_yahoo_gold()
    assert list(df.columns) == ["gold_price"]
    assert df.index.tz == timezone.utc
    assert df.iloc[0]["gold_price"] == 5


@pytest.mark.asyncio
async def test_fetch_yahoo_gold_invalid(monkeypatch):
    def fake_download(*args, **kwargs):
        df = pd.DataFrame({"Open": [1]}, index=[pd.Timestamp("2024-01-01")])
        df.index.name = "Date"
        return df

    monkeypatch.setattr(ingest.yf, "download", fake_download)

    df = await ingest._fetch_yahoo_gold()
    assert list(df.columns) == ["gold_price"]
    assert df.empty
    assert df.index.tz == timezone.utc


@pytest.mark.asyncio
async def test_fetch_yahoo_gold_prices_missing(monkeypatch):
    fixed = pd.Timestamp("2024-01-01", tz="UTC")

    def fake_download(*args, **kwargs):
        raise ingest.YFPricesMissingError("GC=F", {})

    monkeypatch.setattr(pd.Timestamp, "utcnow", staticmethod(lambda: fixed))
    monkeypatch.setattr(ingest.yf, "download", fake_download)

    df = await ingest._fetch_yahoo_gold()
    assert list(df.columns) == ["gold_price"]
    assert df.iloc[0]["gold_price"] is pd.NA
    assert df.index[0] == fixed


@pytest.mark.asyncio
async def test_fetch_yahoo_btc_prices_missing(monkeypatch):
    fixed = pd.Timestamp("2024-01-01", tz="UTC")

    def fake_download(*args, **kwargs):
        raise ingest.YFPricesMissingError("BTC-USD", {})

    monkeypatch.setattr(pd.Timestamp, "utcnow", staticmethod(lambda: fixed))
    monkeypatch.setattr(ingest.yf, "download", fake_download)

    df = await ingest._fetch_yahoo_btc()
    assert list(df.columns) == ["close_usd", "volume"]
    assert df.iloc[0]["close_usd"] is pd.NA
    assert df.iloc[0]["volume"] is pd.NA
    assert df.index[0] == fixed


@pytest.mark.asyncio
async def test_yahoo_helper_squeezes_dataframe(monkeypatch):
    idx = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    cols = pd.MultiIndex.from_product([["Adj Close"], ["GC=F", "OTHER"]])
    df_raw = pd.DataFrame([[4, 5], [6, 7]], index=idx, columns=cols)
    df_raw.index.name = "Date"

    def fake_download(*args, **kwargs):
        return df_raw

    monkeypatch.setattr(ingest.yf, "download", fake_download)

    df = await ingest._fetch_yahoo_gold()
    assert list(df.columns) == ["gold_price"]
    assert df.iloc[0]["gold_price"] == 6

    cols_btc = pd.MultiIndex.from_product([["Adj Close"], ["BTC-USD", "OTHER"]])
    df_btc = pd.DataFrame([[1, 2], [3, 4]], index=idx, columns=cols_btc)
    df_btc.index.name = "Date"

    def fake_download_btc(*args, **kwargs):
        return df_btc

    monkeypatch.setattr(ingest.yf, "download", fake_download_btc)

    df_b = await ingest._fetch_yahoo_btc()
    assert list(df_b.columns) == ["close_usd", "volume"]
    assert df_b.iloc[0]["close_usd"] == 3


@pytest.mark.asyncio
async def test_fred_fallback_to_yahoo(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError("error", request=None, response=self)

    class FakeClient:
        async def get(self, url, timeout=30):
            return FakeResponse()

    async def fake_yahoo():
        df = pd.DataFrame(
            {"gold_price": [7]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
        )
        return df

    monkeypatch.setattr(ingest, "_fetch_yahoo_gold", fake_yahoo)

    df = await ingest._fetch_fred_series(FakeClient(), "GOLDAMGBD228NLBM")
    assert list(df.columns) == ["gold_price"]
    assert df.iloc[0]["gold_price"] == 7


@pytest.mark.asyncio
async def test_fred_fallback_to_yahoo_request_error(monkeypatch):
    class FakeClient:
        async def get(self, url, timeout=30):
            raise httpx.RequestError("boom", request=httpx.Request("GET", url))

    async def fake_yahoo():
        df = pd.DataFrame(
            {"gold_price": [7]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
        )
        return df

    monkeypatch.setattr(ingest, "_fetch_yahoo_gold", fake_yahoo)

    df = await ingest._fetch_fred_series(FakeClient(), "GOLDAMGBD228NLBM")
    assert list(df.columns) == ["gold_price"]
    assert df.iloc[0]["gold_price"] == 7


@pytest.mark.asyncio
async def test_fetch_fred_series_request_error_gold_fallback(monkeypatch):
    """RequestError should trigger Yahoo fallback for the gold series."""

    class FakeClient:
        async def get(self, url, timeout=30):
            raise httpx.RequestError("boom", request=httpx.Request("GET", url))

    async def fake_yahoo():
        return pd.DataFrame(
            {"gold_price": [5]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
        )

    monkeypatch.setattr(ingest, "_fetch_yahoo_gold", fake_yahoo)

    df = await ingest._fetch_fred_series(FakeClient(), "GOLDAMGBD228NLBM")
    assert list(df.columns) == ["gold_price"]
    assert df.iloc[0]["gold_price"] == 5


@pytest.mark.asyncio
async def test_retry_on_429(monkeypatch):
    week_start = pd.Timestamp(datetime.now(tz=timezone.utc))

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, *args, **kwargs):
            self.calls += 1

            class Dummy:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {}

            return Dummy()

    async def fake_coinmetrics(*args, **kwargs):
        return pd.DataFrame({"realised_price": [1], "nupl": [1]}, index=[week_start])

    async def fake_fred(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        return pd.DataFrame({col: [1]}, index=[week_start])

    yahoo_calls = 0

    async def fake_yahoo(*args, **kwargs):
        nonlocal yahoo_calls
        yahoo_calls += 1
        return pd.DataFrame({"close_usd": [7], "volume": [pd.NA]}, index=[week_start])

    async def fake_sleep(_):
        pass

    client = FakeClient()
    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setattr(ingest, "_fetch_yahoo_btc", fake_yahoo)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fred)
    monkeypatch.setattr(ingest.asyncio, "sleep", fake_sleep)

    await ingest.ingest_weekly(week_anchor=week_start)
    assert client.calls == 0
    assert yahoo_calls == 1


@pytest.mark.asyncio
async def test_coingecko_fallback_to_yahoo(monkeypatch):
    week_start = pd.Timestamp(datetime.now(tz=timezone.utc))

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, *args, **kwargs):
            self.calls += 1

            class Dummy:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {}

            return Dummy()

    yahoo_calls = 0

    async def fake_yahoo_btc(*args, **kwargs):
        nonlocal yahoo_calls
        yahoo_calls += 1
        return pd.DataFrame({"close_usd": [7], "volume": [pd.NA]}, index=[week_start])

    async def fake_coinmetrics(*args, **kwargs):
        return pd.DataFrame({"realised_price": [1], "nupl": [1]}, index=[week_start])

    async def fake_fred(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        return pd.DataFrame({col: [1]}, index=[week_start])

    async def fake_sleep(_):
        pass

    client = FakeClient()
    monkeypatch.setattr(ingest.httpx, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setattr(ingest, "_fetch_yahoo_btc", fake_yahoo_btc)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fred)
    monkeypatch.setattr(ingest.asyncio, "sleep", fake_sleep)

    df = await ingest.ingest_weekly(week_anchor=week_start)
    assert client.calls == 0
    assert yahoo_calls == 1
    assert df.loc[0, "close_usd"] == 7


@pytest.mark.asyncio
async def test_ingest_weekly_db_upsert(monkeypatch):
    """ingest_weekly should pass a row dict to execute_values."""

    week_start = pd.Timestamp("2024-01-01", tz="UTC")

    async def fake_fetch_coingecko(client, *args, **kwargs):
        return pd.DataFrame({"close_usd": [10], "volume": [1]}, index=[week_start])

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        df = pd.DataFrame({"realised_price": [1], "nupl": [1]}, index=[week_start])
        return df

    async def fake_fetch_fred_series(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        df = pd.DataFrame({col: [1]}, index=[week_start])
        return df

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)

    captured = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql):
            pass

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def commit(self):
            pass

        def close(self):
            pass

    def fake_connect(url):
        return FakeConn()

    def fake_execute_values(cur, sql, argslist, template=None):
        captured["argslist"] = argslist
        captured["template"] = template

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setattr(ingest.psycopg2, "connect", fake_connect)
    monkeypatch.setattr(ingest.psycopg2.extras, "execute_values", fake_execute_values)

    await ingest.ingest_weekly()

    assert isinstance(captured.get("argslist"), list)
    assert captured["argslist"] and isinstance(captured["argslist"][0], dict)
    assert isinstance(captured.get("template"), str) and captured["template"]
    assert "%(week_start)s" in captured["template"]


@pytest.mark.asyncio
async def test_table_setup_called(monkeypatch):
    calls = []

    async def fake_fetch_coingecko(client, *args, **kwargs):
        return pd.DataFrame(
            {"close_usd": [1], "volume": [1]}, index=[pd.Timestamp.utcnow()]
        )

    async def fake_fetch_coinmetrics(client, *args, **kwargs):
        df = pd.DataFrame(
            {"realised_price": [1], "nupl": [1]}, index=[pd.Timestamp.utcnow()]
        )
        return df

    async def fake_fetch_fred_series(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        return pd.DataFrame({col: [1]}, index=[pd.Timestamp.utcnow()])

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def commit(self):
            pass

        def close(self):
            pass

    def fake_connect(url):
        return FakeConn()

    def fake_execute_values(cur, sql, argslist, template=None):
        pass

    def fake_setup(conn):
        calls.append(True)

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setattr(ingest.psycopg2, "connect", fake_connect)
    monkeypatch.setattr(ingest.psycopg2.extras, "execute_values", fake_execute_values)
    monkeypatch.setattr(ingest, "_create_table_if_missing", fake_setup)

    await ingest.ingest_weekly()

    assert calls == [True]


def test_create_table_if_missing(monkeypatch):
    executed = []

    class FakeCursor:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql):
            executed.append(sql)
            self.calls += 1
            if self.calls == 2:
                raise psycopg2.Error()

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

    ingest._create_table_if_missing(FakeConn())

    assert any("CREATE TABLE IF NOT EXISTS btc_weekly" in sql for sql in executed)


@pytest.mark.asyncio
async def test_fetch_with_retry_request_error_empty_df(monkeypatch):
    calls = 0

    async def failing():
        nonlocal calls
        calls += 1
        raise httpx.RequestError("boom", request=httpx.Request("GET", "http://x"))

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(ingest.asyncio, "sleep", fake_sleep)

    df = await ingest._fetch_with_retry(
        failing, name="coinmetrics", columns=["realised_price", "nupl"]
    )

    assert calls == 3
    assert list(df.columns) == ["realised_price", "nupl"]
    assert df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz == timezone.utc


@pytest.mark.asyncio
async def test_fetch_with_retry_request_error_fallback(monkeypatch):
    calls = 0
    fallback_called = False

    async def failing():
        nonlocal calls
        calls += 1
        raise httpx.RequestError("boom", request=httpx.Request("GET", "http://x"))

    async def fallback():
        nonlocal fallback_called
        fallback_called = True
        return pd.DataFrame(
            {"close_usd": [1]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
        )

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(ingest.asyncio, "sleep", fake_sleep)

    df = await ingest._fetch_with_retry(
        failing,
        name="coingecko",
        columns=["close_usd"],
        fallback=fallback,
    )

    assert calls == 3
    assert fallback_called
    assert not df.empty
    assert df.iloc[0]["close_usd"] == 1


@pytest.mark.asyncio
async def test_fetch_with_retry_read_error(monkeypatch):
    """_fetch_with_retry should return empty DataFrame on ReadError."""

    async def failing():
        raise httpx.ReadError("boom", request=httpx.Request("GET", "http://x"))

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(ingest, "_fetch_coinmetrics", failing)
    monkeypatch.setattr(ingest.asyncio, "sleep", fake_sleep)

    df = await ingest._fetch_with_retry(
        ingest._fetch_coinmetrics,
        name="coinmetrics",
        columns=["realised_price", "nupl"],
    )

    assert list(df.columns) == ["realised_price", "nupl"]
    assert df.empty


@pytest.mark.asyncio
async def test_ingest_weekly_historical_no_data(monkeypatch, caplog):
    week_start = pd.Timestamp("2013-01-07", tz="UTC")

    async def fake_fetch_coingecko(*args, **kwargs):
        return pd.DataFrame(
            {"close_usd": [pd.NA], "volume": [pd.NA]}, index=[week_start]
        )

    async def fake_fetch_coinmetrics(*args, **kwargs):
        return pd.DataFrame(
            {"realised_price": [pd.NA], "nupl": [pd.NA]}, index=[week_start]
        )

    async def fake_fetch_fred_series(client, series_id):
        col = ingest.FRED_COLUMN_MAP.get(series_id, series_id.lower())
        return pd.DataFrame({col: [pd.NA]}, index=[week_start])

    monkeypatch.setattr(ingest, "_fetch_coingecko", fake_fetch_coingecko)
    monkeypatch.setattr(ingest, "_fetch_coinmetrics", fake_fetch_coinmetrics)
    monkeypatch.setattr(ingest, "_fetch_fred_series", fake_fetch_fred_series)

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    called = []

    def fake_init_db(conn, row):
        called.append(True)

    monkeypatch.setattr(ingest, "_init_db", fake_init_db)

    with caplog.at_level(logging.INFO):
        df = await ingest.ingest_weekly(week_anchor=week_start)

    assert len(df) == 1
    assert not called
    assert any("Skipping" in r.message for r in caplog.records)

@pytest.mark.asyncio
async def test_coinmetrics_missing_time(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"error": "Rate limit exceeded"}

    class FakeClient:
        async def get(self, url, params=None, timeout=30):
            return FakeResponse()

    df = await ingest._fetch_coinmetrics(FakeClient())
    assert list(df.columns) == ["realised_price", "nupl"]
    assert df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz == timezone.utc
