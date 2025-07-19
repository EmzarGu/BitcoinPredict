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

# Mapping of FRED series IDs to dataframe column names
FRED_COLUMN_MAP = {
    "WALCL": "fed_liq",
    "ECBASSETS": "ecb_liq",
    "DTWEXBGS": "dxy",
    "DGS10": "ust10",
    "GOLDAMGBD228NLBM": "gold_price",
    "SP500": "spx_index",
}


async def _fetch_coingecko(client: httpx.AsyncClient, days: int = 8) -> Dict[str, Any]:
    resp = await client.get(
        COINGECKO_URL,
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        timeout=30,
    )
    resp.raise_for_status()
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
    client: httpx.AsyncClient, days: int = 8
) -> pd.DataFrame:
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


async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str) -> pd.DataFrame:
    url = FRED_URL.format(series_id=series_id)
    column_name = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        logger.warning("Failed to fetch FRED series %s", series_id)
        return pd.DataFrame(columns=[column_name])
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


async def ingest_weekly() -> pd.DataFrame:
    """Fetch raw data for the current ISO week and aggregate to weekly.

    Missing values are forward-filled so the returned dataframe always has one
    row for the current week. The function never raises and will skip the
    database step if ``DATABASE_URL`` is not configured.

    Returns:
        pd.DataFrame: One-row DataFrame for the current week.
    """

    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    async with httpx.AsyncClient(follow_redirects=True) as client:
        cg_task = asyncio.create_task(_fetch_coingecko(client))
        cm_task = asyncio.create_task(_fetch_coinmetrics(client))
        fed_liq_task = asyncio.create_task(_fetch_fred_series(client, "WALCL"))
        ecb_liq_task = asyncio.create_task(_fetch_fred_series(client, "ECBASSETS"))
        dxy_task = asyncio.create_task(_fetch_fred_series(client, "DTWEXBGS"))
        ust10_task = asyncio.create_task(_fetch_fred_series(client, "DGS10"))
        gold_task = asyncio.create_task(
            _fetch_fred_series(client, "GOLDAMGBD228NLBM")
        )
        sp500_task = asyncio.create_task(_fetch_fred_series(client, "SP500"))

        cg_data = _coingecko_to_weekly(await cg_task)
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
        columns = ",".join(SCHEMA_COLUMNS)
        update = ",".join([f"{c} = EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:]])
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO btc_weekly ({columns}) VALUES %s ON CONFLICT (week_start) DO UPDATE SET {update}",
                [row.iloc[0].to_dict()],
            )
        conn.close()

    return row


if __name__ == "__main__":
    asyncio.run(ingest_weekly())
