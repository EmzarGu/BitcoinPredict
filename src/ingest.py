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
    "DTWEXBGS": "dxy",
    "DGS10": "ust10",
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
    return df[["realised_price", "nupl"]]


async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str) -> pd.DataFrame:
    url = FRED_URL.format(series_id=series_id)
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.lower() for c in df.columns]
    column_name = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    # Rename the first column to "date" since FRED uses "observation_date"
    df.rename(columns={df.columns[0]: "date", df.columns[1]: column_name}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = (
        df.set_index("date")
        .resample("W-MON", label="left", closed="left")
        .last()
    )
    return df[[column_name]]


async def ingest_weekly() -> pd.DataFrame:
    """Fetch raw data for the current ISO week, aggregate to weekly,
    and upsert/insert into the TimescaleDB hypertable `btc_weekly`.

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
        dxy_task = asyncio.create_task(_fetch_fred_series(client, "DTWEXBGS"))
        ust10_task = asyncio.create_task(_fetch_fred_series(client, "DGS10"))
        gold_task = asyncio.create_task(
            _fetch_fred_series(client, "GOLDAMGBD228NLBM")
        )
        sp500_task = asyncio.create_task(_fetch_fred_series(client, "SP500"))

        cg_data = _coingecko_to_weekly(await cg_task)
        cm_data = await cm_task
        fed_liq = await fed_liq_task
        dxy = await dxy_task
        ust10 = await ust10_task
        gold = await gold_task
        sp500 = await sp500_task
    frames = [cg_data, cm_data, fed_liq, dxy, ust10]
    df = pd.concat(frames, axis=1)
    if "volume" in df.columns:
        df = df.drop(columns=["volume"])
    row = df.loc[[week_start]] if week_start in df.index else df.tail(1)
    row = row.copy().reset_index().rename(columns={"index": "week_start"})

    for col in SCHEMA_COLUMNS:
        if col not in row.columns:
            row[col] = pd.NA
    row = row[SCHEMA_COLUMNS]

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set; skipping DB upsert")
    else:
        conn = psycopg2.connect(database_url)
        values = [tuple(row.iloc[0])]
        columns = ",".join(SCHEMA_COLUMNS)
        placeholders = ",".join([f"%({c})s" for c in SCHEMA_COLUMNS])
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
