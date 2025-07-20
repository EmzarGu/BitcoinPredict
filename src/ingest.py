import asyncio
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS: List[str] = [
    "week_start",
    "close_usd",
    "realised_price",
    "nupl",
    "fed_liq",
    "ecb_liq",
    "dxy",
    "ust10",
    "gold_price",
    "spx_index",
]

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_API_KEY = os.getenv("FRED_API_KEY")


# Mapping of FRED series IDs to dataframe column names
FRED_COLUMN_MAP = {
    "WALCL": "fed_liq",
    "ECBASSETS": "ecb_liq",
    "DTWEXBGS": "dxy",
    "DGS10": "ust10",
    "GOLDAMGBD228NLBM": "gold_price",
    "SP500": "spx_index",
}


async def _fetch_coingecko(
    client: httpx.AsyncClient,
    days: int = 8,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Dict[str, Any]:
    """Fetch Bitcoin prices from Coingecko.

    When ``start`` and ``end`` are provided the range endpoint is used. This
    allows retrieval of historical data for backfilling.
    """

    if start or end:
        if not (start and end):
            raise ValueError("Both start and end must be provided for range fetch")
        url = COINGECKO_URL + "/range"
        params = {
            "vs_currency": "usd",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        }
    else:
        url = COINGECKO_URL
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    resp = await client.get(url, params=params, timeout=30)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 429):
            logger.warning("Coingecko HTTP %s - falling back to Yahoo Finance", status)
            return await _fetch_yahoo_btc(start=start, end=end)
        raise
    return resp.json()


def _coingecko_to_weekly(data: Dict[str, Any]) -> pd.DataFrame:
    prices = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])
    vols = pd.DataFrame(data.get("total_volumes", []), columns=["ts", "volume"])
    prices["date"] = pd.to_datetime(prices["ts"], unit="ms", utc=True).dt.floor("D")
    vols["date"] = pd.to_datetime(vols["ts"], unit="ms", utc=True).dt.floor("D")
    df = prices.merge(vols[["date", "volume"]], on="date", how="left")
    df = (
        df.set_index("date")
        .resample("W-MON", label="left", closed="left")
        .agg({"price": "last", "volume": "sum"})
    )
    logger.info("Fetched %s rows for coingecko", len(df))
    if df.empty:
        stub = pd.DataFrame(
            {"close_usd": [pd.NA], "volume": [pd.NA]},
            index=[pd.Timestamp.utcnow().normalize()],
        )
        return stub
    return df.rename(columns={"price": "close_usd"})


async def _fetch_coinmetrics(
    client: httpx.AsyncClient,
    days: int = 8,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Fetch metrics from CoinMetrics.

    If ``start`` and ``end`` are given they delimit the requested range;
    otherwise the last ``days`` are returned.
    """

    if start or end:
        if not (start and end):
            raise ValueError("Both start and end must be provided for range fetch")
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

    params = {
        "assets": "btc",
        "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
    }
    resp = await client.get(COINMETRICS_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    df = pd.DataFrame(data)
    for col in ["CapRealUSD", "SplyCur", "CapMrktCurUSD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("date").resample("W-MON", label="left", closed="left").last()
    df["realised_price"] = df["CapRealUSD"] / df["SplyCur"]
    df["nupl"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
    logger.info("Fetched %s rows for CoinMetrics", len(df))
    if df.empty:
        stub = pd.DataFrame(
            {"realised_price": [pd.NA], "nupl": [pd.NA]},
            index=[pd.Timestamp.utcnow().normalize()],
        )
        return stub
    return df[["realised_price", "nupl"]]




async def _fetch_yahoo_gold() -> pd.DataFrame:
    """Fetch daily gold price from Yahoo Finance and resample to weekly."""
    try:
        raw = await asyncio.to_thread(
            yf.download,
            "GC=F",
            period="1mo",
            interval="1d",
            auto_adjust=False,
        )
    except Exception:
        logger.warning("Failed to fetch gold price from Yahoo Finance")
        empty_index = pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(columns=["gold_price"], index=empty_index)

    # Flatten MultiIndex columns from yfinance (e.g. ('Adj Close', 'GC=F'))
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    price_col = None
    if "Adj Close" in raw.columns:
        price_col = "Adj Close"
    elif "Close" in raw.columns:
        price_col = "Close"
    if price_col is None:
        logger.warning("Yahoo Finance data missing Close column")
        empty_index = pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(columns=["gold_price"], index=empty_index)

    series = raw[price_col].rename("gold_price")
    series.index = pd.to_datetime(series.index, utc=True)
    df = (
        series.to_frame()
        .resample("W-MON", label="left", closed="left")
        .last()
    )
    logger.info("Fetched %s rows for Yahoo gold price", len(df))
    if df.empty:
        stub = pd.DataFrame(
            {"gold_price": [pd.NA]}, index=[pd.Timestamp.utcnow().normalize()]
        )
        return stub
    return df[["gold_price"]]


async def _fetch_yahoo_btc(start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """Fetch daily BTC price from Yahoo Finance and resample to weekly."""
    kwargs = {"interval": "1d", "auto_adjust": False}
    if start and end:
        kwargs.update({"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")})
    else:
        kwargs.update({"period": "1mo"})
    try:
        raw = await asyncio.to_thread(yf.download, "BTC-USD", **kwargs)
    except Exception:
        logger.warning("Failed to fetch BTC price from Yahoo Finance")
        empty_index = pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(columns=["close_usd"], index=empty_index)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    price_col = None
    if "Adj Close" in raw.columns:
        price_col = "Adj Close"
    elif "Close" in raw.columns:
        price_col = "Close"
    if price_col is None:
        logger.warning("Yahoo Finance BTC data missing Close column")
        empty_index = pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(columns=["close_usd"], index=empty_index)

    series = raw[price_col].rename("close_usd")
    series.index = pd.to_datetime(series.index, utc=True)
    df = series.to_frame().resample("W-MON", label="left", closed="left").last()
    logger.info("Fetched %s rows for Yahoo BTC price", len(df))
    if df.empty:
        stub = pd.DataFrame({"close_usd": [pd.NA]}, index=[pd.Timestamp.utcnow().normalize()])
        return stub
    return df[["close_usd"]]


async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str) -> pd.DataFrame:
    url = FRED_URL.format(series_id=series_id)
    if FRED_API_KEY:
        url += f"&api_key={FRED_API_KEY}"
    column_name = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        logger.warning("Failed to fetch FRED series %s", series_id)
        if series_id == "GOLDAMGBD228NLBM":
            return await _fetch_yahoo_gold()
        empty_index = pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(columns=[column_name], index=empty_index)
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.lower() for c in df.columns]
    # Rename the first column to "date" since FRED uses "observation_date"
    df.rename(columns={df.columns[0]: "date", df.columns[1]: column_name}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = (
        df.set_index("date")
        .resample("W-MON", label="left", closed="left")
        .last()
    )
    logger.info("Fetched %s rows for %s", len(df), series_id)
    if df.empty:
        stub = pd.DataFrame(
            {column_name: [pd.NA]},
            index=[pd.Timestamp.utcnow().normalize()],
        )
        return stub
    return df[[column_name]]


async def _fetch_with_retry(func, *, name: str, columns: List[str], fallback=None):
    """Retry wrapper for network fetchers."""
    delay = 1
    for attempt in range(3):
        try:
            return await func()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retryable = status >= 500 or (name == "coingecko" and status in (401, 429))
            if not retryable:
                raise
            if attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if name == "coingecko" and status in (401, 429) and fallback:
                return await fallback()
            logger.warning("Failed to fetch %s after retries", name)
            empty_index = pd.DatetimeIndex([], tz="UTC")
            return pd.DataFrame(columns=columns, index=empty_index)


def _create_table_if_missing(conn: psycopg2.extensions.connection) -> None:
    """Create the ``btc_weekly`` table if it doesn't exist.

    This function also attempts to convert the table into a TimescaleDB
    hypertable when the extension is available. Both operations are safe to
    call multiple times.
    """

    create_sql = """
    CREATE TABLE IF NOT EXISTS btc_weekly (
        week_start TIMESTAMPTZ PRIMARY KEY,
        close_usd DOUBLE PRECISION,
        realised_price DOUBLE PRECISION,
        nupl DOUBLE PRECISION,
        fed_liq DOUBLE PRECISION,
        ecb_liq DOUBLE PRECISION,
        dxy DOUBLE PRECISION,
        ust10 DOUBLE PRECISION,
        gold_price DOUBLE PRECISION,
        spx_index DOUBLE PRECISION
    );
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()
    logger.info("btc_weekly table ensured and committed")

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT create_hypertable('btc_weekly','week_start', if_not_exists => TRUE);"
            )
        conn.commit()
    except psycopg2.Error:
        conn.rollback()
        logger.info(
            "Timescale extension unavailable; continuing with plain table"
        )


def _init_db(conn: psycopg2.extensions.connection, row: Dict[str, Any]) -> None:
    """Ensure table exists and upsert a single row."""

    _create_table_if_missing(conn)

    columns = ",".join(SCHEMA_COLUMNS)
    update = ",".join([f"{c} = EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:]])
    with conn.cursor() as cur:
        template = "(" + ",".join(f"%({col})s" for col in SCHEMA_COLUMNS) + ")"
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO btc_weekly ({columns}) VALUES %s ON CONFLICT (week_start) DO UPDATE SET {update}",
            [row],
            template=template,
        )
    conn.commit()



async def ingest_weekly(week_anchor: datetime | None = None) -> pd.DataFrame:
    """Fetch raw data for the specified ISO week and aggregate to weekly.

    When ``week_anchor`` is ``None`` the current week is used. Missing values are
    forward-filled so the returned dataframe always has one row. The function
    never raises and will skip the database step if ``DATABASE_URL`` is not
    configured.

    Returns:
        pd.DataFrame: One-row DataFrame for the requested week.
    """

    now = week_anchor or datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    async with httpx.AsyncClient(follow_redirects=True) as client:
        if week_anchor:
            cg_task = asyncio.create_task(
                _fetch_with_retry(
                    lambda: _fetch_coingecko(
                        client,
                        start=week_start,
                        end=week_start + timedelta(days=7),
                    ),
                    name="coingecko",
                    columns=["close_usd", "volume"],
                    fallback=lambda: _fetch_yahoo_btc(
                        start=week_start, end=week_start + timedelta(days=7)
                    ),
                )
            )
            cm_task = asyncio.create_task(
                _fetch_with_retry(
                    lambda: _fetch_coinmetrics(
                        client,
                        start=week_start,
                        end=week_start + timedelta(days=7),
                    ),
                    name="coinmetrics",
                    columns=["realised_price", "nupl"],
                )
            )
        else:
            cg_task = asyncio.create_task(
                _fetch_with_retry(
                    lambda: _fetch_coingecko(client),
                    name="coingecko",
                    columns=["close_usd", "volume"],
                    fallback=lambda: _fetch_yahoo_btc(),
                )
            )
            cm_task = asyncio.create_task(
                _fetch_with_retry(
                    lambda: _fetch_coinmetrics(client),
                    name="coinmetrics",
                    columns=["realised_price", "nupl"],
                )
            )
        fed_liq_task = asyncio.create_task(_fetch_fred_series(client, "WALCL"))
        ecb_liq_task = asyncio.create_task(_fetch_fred_series(client, "ECBASSETS"))
        dxy_task = asyncio.create_task(_fetch_fred_series(client, "DTWEXBGS"))
        ust10_task = asyncio.create_task(_fetch_fred_series(client, "DGS10"))
        gold_task = asyncio.create_task(
            _fetch_fred_series(client, "GOLDAMGBD228NLBM")
        )
        sp500_task = asyncio.create_task(_fetch_fred_series(client, "SP500"))

        cg_raw = await cg_task
        if isinstance(cg_raw, dict):
            cg_data = _coingecko_to_weekly(cg_raw)
        else:
            cg_data = cg_raw
        cm_data = await cm_task
        fed_liq = await fed_liq_task
        ecb_liq = await ecb_liq_task
        dxy = await dxy_task
        ust10 = await ust10_task
        gold = await gold_task
        sp500 = await sp500_task
    frames = [cg_data, cm_data, fed_liq, ecb_liq, dxy, ust10, gold, sp500]
    df = pd.concat(frames, axis=1)
    if "volume" in df.columns:
        df = df.drop(columns=["volume"])
    df = df.sort_index().ffill()
    df = df.infer_objects(copy=False)
    df.index.name = "date"
    df_weekly = (
        df.resample("W-MON", label="left", closed="left").last().reset_index()
    )
    df_weekly["week_start"] = df_weekly["date"].dt.normalize()
    row = df_weekly[df_weekly["week_start"] == week_start]
    if row.empty:
        row = df_weekly.tail(1)
    row = row.copy().drop(columns=["date"])

    for col in SCHEMA_COLUMNS:
        if col not in row.columns:
            row[col] = pd.NA
    row = row[SCHEMA_COLUMNS]

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set; skipping DB upsert")
    else:
        conn = psycopg2.connect(database_url)
        _init_db(conn, row.iloc[0].to_dict())
        conn.close()

    return row


if __name__ == "__main__":
    asyncio.run(ingest_weekly())
