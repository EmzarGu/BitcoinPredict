#!/usr/bin/env python3
from __future__ import annotations

# ─── Restore built-in str to avoid any shadowing issues ─────────────────────
import builtins
str = builtins.str
# ─────────────────────────────────────────────────────────────────────────────

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

# ─── Configuration ──────────────────────────────────────────────────────────
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

# ─── Utilities ──────────────────────────────────────────────────────────────
def to_python_float(value: Any) -> float | None:
    if pd.isna(value) or value is None:
        return None
    return float(value)


def get_db_connection() -> psycopg2.extensions.connection:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing in .env")
    return psycopg2.connect(url)


def _create_table_if_missing(conn: psycopg2.extensions.connection) -> None:
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


def _upsert_row(conn: psycopg2.extensions.connection, row: Dict[str, Any]) -> None:
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

# ─── Fetch helpers ──────────────────────────────────────────────────────────
async def _fetch_coingecko(
    client: httpx.AsyncClient, start: datetime, end: datetime
) -> pd.DataFrame:
    params = {"vs_currency": "usd", "days": (end - start).days + 1, "interval": "daily"}
    try:
        r = await client.get(COINGECKO_URL, params=params, timeout=30)
        r.raise_for_status()
        prices = pd.DataFrame(r.json().get("prices", []), columns=["ts", "price"])
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms", utc=True).dt.floor("D")
        df = prices.set_index("date")[['price']].rename(columns={'price':'close_usd'})
        return df
    except httpx.HTTPStatusError as e:
        logger.warning(f"CoinGecko fetch failed ({e}); falling back to Yahoo Finance.")
    except Exception as e:
        logger.warning(f"CoinGecko error ({e}); falling back to Yahoo Finance.")
    # Fallback to Yahoo Finance
    try:
        raw = await asyncio.to_thread(
            yf.download,
            "BTC-USD",
            start=start.strftime('%Y-%m-%d'),
            end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        price_col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
        series = raw[price_col].copy()
        series.name = 'close_usd'
        series.index = pd.to_datetime(series.index, utc=True)
        return series.to_frame()
    except Exception as e:
        logger.warning(f"Yahoo BTC fetch failed: {e}")
    return pd.DataFrame()


async def _fetch_coinmetrics(
    client: httpx.AsyncClient, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
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
    df['date'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('date')
    for c in ['CapRealUSD','SplyCur','CapMrktCurUSD']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['realised_price'] = df['CapRealUSD'] / df['SplyCur']
    df['nupl'] = (df['CapMrktCurUSD'] - df['CapRealUSD']) / df['CapMrktCurUSD']
    return df[['realised_price','nupl']]


async def _fetch_fred_series(
    client: httpx.AsyncClient, series_id: str, start: datetime, end: datetime
) -> pd.DataFrame:
    url = FRED_URL.format(series_id=series_id)
    col = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        df.index = df.index.tz_localize('UTC')
        df.columns = [col]
        df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[[col]].loc[start:end]
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
    return pd.DataFrame()


async def _fetch_yahoo_gold(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch gold price via Yahoo Finance (ticker GC=F) over the full historical window,
    returning a UTC-indexed DataFrame with column 'gold_price'.
    """
    import pandas as _pd
    try:
        # Download including end date
        raw = await asyncio.to_thread(
            yf.download,
            "GC=F",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            logger.warning("Yahoo gold fetch returned empty for GC=F over %s to %s", start, end)
            return _pd.DataFrame()
        # Drop ticker-level if present
        if isinstance(raw.columns, _pd.MultiIndex):
            raw.columns = raw.columns.droplevel(0)
        # Identify the price column
        price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if price_col not in raw.columns:
            logger.warning("Yahoo gold data missing '%s' column", price_col)
            return _pd.DataFrame()
        # Build series
        series = raw[price_col].copy()
        series.name = "gold_price"
        series.index = _pd.to_datetime(series.index).tz_localize("UTC")
        return series.to_frame()
    except Exception as e:
        logger.warning(f"Yahoo gold fetch failed for GC=F: {e}")
    return pd.DataFrame()

# ─── Main ingestion ─────────────────────────────────────────────────────────
# (No changes here; orchestration unchanged)
async def ingest_weekly(week_anchor=None, years=1):
    if week_anchor and week_anchor.tzinfo is None:
        week_anchor = week_anchor.replace(tzinfo=timezone.utc)
    now = week_anchor or datetime.now(timezone.utc)
    end_date = now
    start_date = end_date - timedelta(days=365 * years)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = {
            'btc':    _fetch_coingecko(client, start=start_date, end=end_date),
            'cm':     _fetch_coinmetrics(client, start_date=start_date, end_date=end_date),
            'fed_liq':_fetch_fred_series(client, 'WALCL', start_date, end_date),
            'ecb_liq':_fetch_fred_series(client, 'ECBASSETS', start_date, end_date),
            'dxy':    _fetch_fred_series(client, 'DTWEXBGS', start_date, end_date),
            'ust10':  _fetch_fred_series(client, 'DGS10', start_date, end_date),
            'gold':   _fetch_yahoo_gold(start_date, end_date),
            'spx':    _fetch_fred_series(client, 'SP500', start_date, end_date),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        dfs = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Task {key} failed: {result}")
                dfs[key] = pd.DataFrame()
            else:
                dfs[key] = result

    if dfs['btc'].empty:
        logger.error("Bitcoin price unavailable – aborting ingest.")
        return

    df_all = pd.concat([df for df in dfs.values() if not df.empty], axis=1)
    df_all = df_all.sort_index().ffill()
    df_all.dropna(subset=['close_usd'], inplace=True)
    if df_all.empty:
        logger.warning("No data after merge – aborting.")
        return

    weekly = df_all.resample('W-MON', label='left', closed='left').last()
    rows: List[Dict[str, Any]] = []
    for ts, row in weekly.iterrows():
        rec = {c: None for c in SCHEMA_COLUMNS}
        rec['week_start'] = ts
        for k, v in row.items():
            rec[k] = to_python_float(v)
        rows.append(rec)

    with get_db_connection() as conn:
        _create_table_if_missing(conn)
        for r in rows:
            _upsert_row(conn, r)
        logger.info(f"✅ Upserted {len(rows)} weekly rows.")

# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=args.years))
