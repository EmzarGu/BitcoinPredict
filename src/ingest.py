#!/usr/bin/env python3
from __future__ import annotations

# Restore built-in str
import builtins
str = builtins.str

import asyncio
import os
import io
from datetime import datetime, timedelta, timezone
import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf
from dotenv import load_dotenv
from typing import Any  # <-- Added missing import

# Load environment
load_dotenv()

# Diagnostics

def debug(msg: str):
    print(f"[DEBUG] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

# Schema
SCHEMA_COLUMNS: list[str] = [
    "week_start", "close_usd", "realised_price", "nupl",
    "fed_liq", "ecb_liq", "dxy", "ust10",
    "gold_price", "spx_index",
]

# External endpoints
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_MAP: dict[str, str] = {
    "WALCL": "fed_liq",
    "ECBASSETS": "ecb_liq",
    "DTWEXBGS": "dxy",
    "DGS10": "ust10",
    "SP500": "spx_index",
}

# Database helpers

def get_db_connection():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing in .env")
    return psycopg2.connect(url)

def _create_table_if_missing(conn):
    sql = '''
    CREATE TABLE IF NOT EXISTS btc_weekly (
        week_start TIMESTAMPTZ PRIMARY KEY,
        close_usd REAL, realised_price REAL, nupl REAL,
        fed_liq REAL, ecb_liq REAL, dxy REAL, ust10 REAL,
        gold_price REAL, spx_index REAL
    );
    '''
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def _upsert_row(conn, row: dict[str, Any]):
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

# Fetch helpers

async def _fetch_yahoo_gold(start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Fetching gold from GC=F: {start} to {end}")
    try:
        raw = await asyncio.to_thread(
            yf.download,
            "GC=F",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        debug(f"Raw shape: {raw.shape}")
        if raw.empty:
            warn("Yahoo gold: no data returned")
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(0)  # fixed level
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col not in raw.columns:
            warn(f"Yahoo gold missing column '{col}'")
            return pd.DataFrame()
        series = raw[col].copy()
        series.name = "gold_price"
        # Use utc=True to avoid tz_localize errors
        series.index = pd.to_datetime(series.index, utc=True)
        debug(f"Series shape: {series.shape}")
        return series.to_frame()
    except Exception as e:
        warn(f"Error fetching gold: {e}")
        return pd.DataFrame()

async def _fetch_coingecko(client: httpx.AsyncClient, start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Fetching BTC from CoinGecko: {start} to {end}")
    try:
        params = {"vs_currency": "usd", "days": (end - start).days + 1, "interval": "daily"}
        r = await client.get(COINGECKO_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("prices", [])
        df = pd.DataFrame(data, columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("D")
        df = df.set_index("date")[['price']].rename(columns={'price':'close_usd'})
        debug(f"CoinGecko DF shape: {df.shape}")
        return df
    except Exception as e:
        warn(f"CoinGecko error: {e}")
    # fallback to Yahoo
    debug("Falling back to Yahoo for BTC")
    raw = await asyncio.to_thread(
        yf.download,
        "BTC-USD",
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    debug(f"Yahoo BTC raw shape: {raw.shape}")
    if raw.empty:
        warn("Yahoo BTC: no data returned")
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(0)  # fixed level
    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    if col not in raw.columns:
        warn(f"Yahoo BTC missing column '{col}'")
        return pd.DataFrame()
    series = raw[col].copy()
    series.name = "close_usd"
    series.index = pd.to_datetime(series.index, utc=True)
    df = series.to_frame()
    debug(f"BTC series shape: {df.shape}")
    return df

async def _fetch_coinmetrics(client: httpx.AsyncClient, start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Fetching CoinMetrics: {start} to {end}")
    params = {
        "assets": "btc",
        "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
    }
    r = await client.get(COINMETRICS_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        warn("CoinMetrics: no data")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('date')
    for c in ['CapRealUSD','SplyCur','CapMrktCurUSD']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['realised_price'] = df['CapRealUSD'] / df['SplyCur']
    df['nupl'] = (df['CapMrktCurUSD'] - df['CapRealUSD']) / df['CapMrktCurUSD']
    debug(f"CoinMetrics DF shape: {df.shape}")
    return df[['realised_price','nupl']]

async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Fetching FRED {series_id}: {start} to {end}")
    try:
        r = await client.get(FRED_URL.format(series_id=series_id), timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
        # Use utc=True to avoid tz_localize errors
        df.index = pd.to_datetime(df.index, utc=True)
        col = FRED_MAP.get(series_id, series_id.lower())
        df.columns = [col]
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[[col]].loc[start:end]
        debug(f"FRED DF shape: {df.shape}")
        return df
    except Exception as e:
        warn(f"FRED {series_id} error: {e}")
        return pd.DataFrame()

# Main ingestion
async def ingest_weekly(week_anchor=None, years: int = 1):
    debug(f"Starting ingest_weekly with anchor={week_anchor}, years={years}")
    if isinstance(week_anchor, datetime) and week_anchor.tzinfo is None:
        week_anchor = week_anchor.replace(tzinfo=timezone.utc)
    elif isinstance(week_anchor, datetime.date):
        week_anchor = datetime.combine(week_anchor, datetime.min.time(), tzinfo=timezone.utc)
    now = week_anchor or datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    debug(f"Window: {start} to {now}")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = {
            'btc': _fetch_coingecko(client, start, now),
            'cm': _fetch_coinmetrics(client, start, now),
            'fed_liq': _fetch_fred_series(client, 'WALCL', start, now),
            'ecb_liq': _fetch_fred_series(client, 'ECBASSETS', start, now),
            'dxy': _fetch_fred_series(client, 'DTWEXBGS', start, now),
            'ust10': _fetch_fred_series(client, 'DGS10', start, now),
            'gold': _fetch_yahoo_gold(start, now),
            'spx': _fetch_fred_series(client, 'SP500', start, now),
        }
        debug(f"Launching tasks: {list(tasks.keys())}")
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        dfs = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                warn(f"Task {key} error: {result}")
                dfs[key] = pd.DataFrame()
            else:
                dfs[key] = result
                debug(f"{key} returned {result.shape}")

    df_all = pd.concat([df for df in dfs.values() if not df.empty], axis=1)
    debug(f"Merged DF shape: {df_all.shape}")
    df_all = df_all.sort_index().ffill()
    df_all.dropna(subset=['close_usd'], inplace=True)
    debug(f"Cleaned DF shape: {df_all.shape}")
    if df_all.empty:
        warn("No data to upsert")
        return

    weekly = df_all.resample('W-MON', label='left', closed='left').last()
    debug(f"Weekly DF shape: {weekly.shape}")

    conn = get_db_connection()
    _create_table_if_missing(conn)
    count = 0
    for ts, row in weekly.iterrows():
        rec = {'week_start': ts}
        for col, val in row.items():
            rec[col] = None if pd.isna(val) else float(val)
        for col in SCHEMA_COLUMNS:
            rec.setdefault(col, None)
        _upsert_row(conn, rec)
        count += 1
    debug(f"Upserted {count} rows")

# CLI
if __name__ == '__main__':
    asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=1))
