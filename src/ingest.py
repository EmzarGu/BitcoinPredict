#!/usr/bin/env python3
from __future__ import annotations

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
import yfinance as yf
from dotenv import load_dotenv

# ─── Config ────────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SCHEMA_COLUMNS: List[str] = [
    "week_start", "close_usd", "realised_price", "nupl",
    "fed_liq", "ecb_liq", "dxy", "ust10",
    "gold_price", "spx_index",
]

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FRED_COLUMN_MAP = {
    "WALCL": "fed_liq",
    "ECBASSETS": "ecb_liq",
    "DTWEXBGS": "dxy",
    "DGS10": "ust10",
    "SP500": "spx_index",
}

# ─── Utility ───────────────────────────────────────────────────────────────
def to_python_float(v):
    if pd.isna(v) or v is None:
        return None
    return float(v)

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL missing in .env")
    return psycopg2.connect(db_url)

def _create_table_if_missing(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS btc_weekly (
        week_start TIMESTAMPTZ PRIMARY KEY,
        close_usd REAL, realised_price REAL, nupl REAL,
        fed_liq REAL, ecb_liq REAL, dxy REAL, ust10 REAL,
        gold_price REAL, spx_index REAL
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def _upsert_row(conn, row: Dict[str, Any]):
    cols = ",".join(SCHEMA_COLUMNS)
    updates = ",".join(f"{c}=EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:])
    tmpl = "(" + ",".join(f"%({c})s" for c in SCHEMA_COLUMNS) + ")"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO btc_weekly ({cols}) VALUES %s "
            f"ON CONFLICT (week_start) DO UPDATE SET {updates}",
            [row],
            template=tmpl,
        )
    conn.commit()

# ─── Data fetch helpers (all your originals) ───────────────────────────────
async def _fetch_coingecko(client, start, end):
    params = {
        "vs_currency": "usd",
        "days": int((end - start).days) + 1,
        "interval": "daily",
    }
    r = await client.get(COINGECKO_URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["prices"], columns=["ts", "close_usd"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("D")
    return df.set_index("date")[["close_usd"]]

async def _fetch_coinmetrics(client, start_date, end_date):
    params = {
        "assets": "btc",
        "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency": "1d",
        "start_time": start_date.strftime("%Y-%m-%d"),
        "end_time": end_date.strftime("%Y-%m-%d"),
    }
    r = await client.get(COINMETRICS_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("date")
    for c in ("CapRealUSD", "SplyCur", "CapMrktCurUSD"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["realised_price"] = df["CapRealUSD"] / df["SplyCur"]
    df["nupl"] = (
        (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
    )
    return df[["realised_price", "nupl"]]

async def _fetch_fred_series(client, series_id, start, end):
    url = FRED_URL.format(series_id=series_id)
    col = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    try:
        r = await client.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
        df.index = df.index.tz_localize("UTC")
        df.columns = [col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[[col]].loc[start:end]
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
    return pd.DataFrame()

# ─── NEW: dedicated gold fetcher ───────────────────────────────────────────
async def _fetch_gold(start, end):
    """Use Yahoo Finance first; if that fails, fall back to FRED."""
    try:
        raw = await asyncio.to_thread(
            yf.download, "GC=F",
            start=start, end=end + timedelta(days=1),
            auto_adjust=True, progress=False
        )
        if not raw.empty:
            price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            s = raw[price_col]
            s.index = s.index.tz_localize("UTC")
            return s.rename("gold_price").to_frame()
    except Exception as e:
        logger.warning(f"Yahoo gold fetch failed: {e}")

    # Fallback – same CSV endpoint you used originally
    async with httpx.AsyncClient() as client:
        fred_df = await _fetch_fred_series(
            client, "GOLDAMGBD228NLBM", start, end
        )
    return fred_df

# ─── Main ingest coroutine (unchanged except gold task) ────────────────────
async def ingest_weekly(week_anchor=None, years=1):
    now = week_anchor or datetime.now(timezone.utc)
    end_date = now
    start_date = now - timedelta(days=365 * years)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = {
            "btc": _fetch_coingecko(client, start=start_date, end=end_date),
            "cm": _fetch_coinmetrics(client, start_date=start_date, end_date=end_date),
            "fed": _fetch_fred_series(client, "WALCL", start_date, end_date),
            "ecb": _fetch_fred_series(client, "ECBASSETS", start_date, end_date),
            "dxy": _fetch_fred_series(client, "DTWEXBGS", start_date, end_date),
            "ust10": _fetch_fred_series(client, "DGS10", start_date, end_date),
            "gold": _fetch_gold(start_date, end_date),              # ← changed
            "spx": _fetch_fred_series(client, "SP500", start_date, end_date),
        }
        results = await asyncio.gather(*tasks.values())
        dfs = dict(zip(tasks.keys(), results))

    if dfs["btc"].empty:
        logger.error("Bitcoin price unavailable – aborting ingest.")
        return

    merged = pd.concat([df for df in dfs.values() if not df.empty], axis=1)
    merged = merged.sort_index().ffill()
    merged.dropna(subset=["close_usd"], inplace=True)
    if merged.empty:
        logger.warning("Merged dataframe empty – nothing to insert.")
        return

    weekly = merged.resample("W-MON", label="left", closed="left").last()
    rows = []
    for ts, row in weekly.iterrows():
        rec = {"week_start": ts}
        rec.update({k: to_python_float(v) for k, v in row.items()})
        for col in SCHEMA_COLUMNS:
            rec.setdefault(col, None)
        rows.append(rec)

    with get_db_connection() as conn:
        _create_table_if_missing(conn)
        for r in rows:
            _upsert_row(conn, r)
    logger.info("✅ Upserted %d weekly rows.", len(rows))

# ─── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=args.years))
