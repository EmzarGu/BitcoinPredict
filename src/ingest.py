#!/usr/bin/env python3
"""
Weekly ETL pipeline for the btc_weekly table.

Key improvement v0.1.1 (2025‑08‑04)
-----------------------------------
• Robust gold-price ingestion:
  – Accept FRED series GOLDAMGBD228NLBM only if ≥40 % of rows
    in the requested range are numeric.
  – Otherwise fall back automatically to Yahoo Finance (ticker GC=F).
• Safe numeric casting helper is reused for every metric.
• Code is reorganised for clarity: fetchers are plain functions,
  orchestration is in `async ingest_weekly`.
"""
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

# ─── Globals ──────────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SCHEMA_COLUMNS: List[str] = [
    "week_start", "close_usd", "realised_price", "nupl", "fed_liq",
    "ecb_liq", "dxy", "ust10", "gold_price", "spx_index",
]

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_MAP = {
    "WALCL": "fed_liq",
    "ECBASSETS": "ecb_liq",
    "DTWEXBGS": "dxy",
    "DGS10": "ust10",
    "GOLDAMGBD228NLBM": "gold_price",
    "SP500": "spx_index",
}

# ─── Utility helpers ──────────────────────────────────────────────────────────
def to_float(x):
    """Cast to python float or return None."""
    return None if pd.isna(x) else float(x)

def get_db():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing – check .env")
    return psycopg2.connect(url)

def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_weekly (
                week_start TIMESTAMPTZ PRIMARY KEY,
                close_usd    REAL, realised_price REAL, nupl REAL,
                fed_liq      REAL, ecb_liq        REAL, dxy  REAL,
                ust10        REAL, gold_price     REAL, spx_index REAL
            );
            """
        )
    conn.commit()

def upsert_rows(conn, rows: List[Dict[str, Any]]):
    cols = ",".join(SCHEMA_COLUMNS)
    sets = ",".join(f"{c}=EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:])
    tmpl = "(" + ",".join(f"%({c})s" for c in SCHEMA_COLUMNS) + ")"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO btc_weekly ({cols}) VALUES %s "
            f"ON CONFLICT (week_start) DO UPDATE SET {sets}",
            rows,
            template=tmpl,
        )
    conn.commit()

# ─── Fetch helpers ────────────────────────────────────────────────────────────
async def fred_series(client, sid: str) -> pd.DataFrame:
    """Return a daily FRED series, numeric; may contain NaN."""
    url = FRED_URL.format(sid=sid)
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    df.index = df.index.tz_localize("UTC")
    col = FRED_MAP[sid]
    df.columns = [col]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[[col]]

async def fetch_btc(client) -> pd.DataFrame:
    """BTC close from CoinGecko; fall back to Yahoo on failure."""
    try:
        p = {"vs_currency": "usd", "days": 8, "interval": "daily"}
        r = await client.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params=p, timeout=30
        )
        r.raise_for_status()
        prices = pd.DataFrame(r.json()["prices"], columns=["ts", "price"])
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms", utc=True).dt.floor("D")
        return prices.set_index("date")[["price"]].rename(columns={"price": "close_usd"})
    except Exception as e:
        logger.warning("CoinGecko failed, fallback to Yahoo: %s", e)

    # Yahoo fallback
    raw = await asyncio.to_thread(yf.download, "BTC-USD", period="10d", progress=False)
    price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    s = raw[price_col]
    s.index = s.index.tz_localize("UTC")
    return s.rename("close_usd").to_frame()

async def fetch_coinmetrics(client, start: datetime, end: datetime) -> pd.DataFrame:
    """Realised price & NUPL — CoinMetrics community API."""
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    p = {
        "assets": "btc",
        "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
    }
    r = await client.get(url, params=p, timeout=30)
    r.raise_for_status()
    data = pd.DataFrame(r.json()["data"])
    if data.empty:
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["time"], utc=True)
    data = data.set_index("date")
    for c in ("CapRealUSD", "SplyCur", "CapMrktCurUSD"):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["realised_price"] = data["CapRealUSD"] / data["SplyCur"]
    data["nupl"] = (data["CapMrktCurUSD"] - data["CapRealUSD"]) / data["CapMrktCurUSD"]
    return data[["realised_price", "nupl"]]

async def fetch_gold(client, start: datetime, end: datetime) -> pd.DataFrame:
    """Gold price daily series with quality check and fallback."""
    fred = await fred_series(client, "GOLDAMGBD228NLBM")
    # restrict to window
    fred = fred.loc[start:end]
    ok_frac = fred["gold_price"].notna().mean()
    if ok_frac >= 0.40:
        logger.info("Using FRED gold series (%.0f %% complete).", ok_frac * 100)
        return fred.ffill()

    logger.warning("FRED gold incomplete (%.0f %% valid) – switching to Yahoo.", ok_frac * 100)
    raw = await asyncio.to_thread(
        yf.download, "GC=F", start=start, end=end + timedelta(days=1),
        auto_adjust=True, progress=False
    )
    if raw.empty:
        logger.error("Yahoo gold price fetch failed; returning empty frame.")
        return pd.DataFrame()

    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    s = raw[col]
    s.index = s.index.tz_localize("UTC")
    return s.rename("gold_price").to_frame()

# ─── Main orchestrator ────────────────────────────────────────────────────────
async def ingest_weekly(anchor: datetime | None = None, years: int = 1) -> None:
    anchor = anchor or datetime.now(timezone.utc)
    start = anchor - timedelta(days=365 * years)
    end   = anchor

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = {
            "btc":    fetch_btc(client),
            "cm":     fetch_coinmetrics(client, start, end),
            "fed":    fred_series(client, "WALCL"),
            "ecb":    fred_series(client, "ECBASSETS"),
            "dxy":    fred_series(client, "DTWEXBGS"),
            "ust10":  fred_series(client, "DGS10"),
            "gold":   fetch_gold(client, start, end),
            "spx":    fred_series(client, "SP500"),
        }
        results = await asyncio.gather(*tasks.values())
        dfs = dict(zip(tasks.keys(), results))

    if dfs["btc"].empty:
        logger.error("Bitcoin price unavailable – aborting ingest.")
        return

    # merge and forward‑fill
    daily = pd.concat([df for df in dfs.values() if not df.empty], axis=1).sort_index()
    daily = daily.ffill().dropna(subset=["close_usd"])  # must have BTC price

    weekly = daily.resample("W-MON", label="left", closed="left").last()
    rows: List[Dict[str, Any]] = []
    for ts, r in weekly.iterrows():
        d: Dict[str, Any] = {c: None for c in SCHEMA_COLUMNS}
        d["week_start"] = ts
        for k, v in r.items():
            d[k] = to_float(v)
        rows.append(d)

    if not rows:
        logger.warning("Nothing new to insert.")
        return

    with get_db() as conn:
        ensure_table(conn)
        upsert_rows(conn, rows)
    logger.info("✅ %d weekly rows upserted.", len(rows))

# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="Ingest weekly market data")
    p.add_argument("--years", type=int, default=1, help="History window (years)")
    args = p.parse_args()
    try:
        asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=args.years))
    except KeyboardInterrupt:
        sys.exit(130)
