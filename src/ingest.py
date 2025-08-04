#!/usr/bin/env python3
from __future__ import annotations

# Restore built-in str
import builtins
str = builtins.str

import asyncio
import io
import os
from datetime import datetime, timedelta, timezone
import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Simple print diagnostics
def debug(msg: str):
    print(f"[DEBUG] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

# Schema columns
SCHEMA_COLUMNS = [
    "week_start", "close_usd", "realised_price", "nupl",
    "fed_liq", "ecb_liq", "dxy", "ust10",
    "gold_price", "spx_index",
]

# External endpoints
COINGECKO_URL   = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL        = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_MAP        = {
    "WALCL":     "fed_liq",
    "ECBASSETS": "ecb_liq",
    "DTWEXBGS":  "dxy",
    "DGS10":     "ust10",
    "SP500":     "spx_index",
}

# Database helpers
def get_db_connection():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing in .env")
    return psycopg2.connect(url)

def _create_table_if_missing(conn):
    create_sql = """
    CREATE TABLE IF NOT EXISTS btc_weekly (
        week_start TIMESTAMPTZ PRIMARY KEY,
        close_usd REAL, realised_price REAL, nupl REAL,
        fed_liq REAL, ecb_liq REAL, dxy REAL, ust10 REAL,
        gold_price REAL, spx_index REAL
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()

def _upsert_row(conn, row: dict[str, Any]):
    cols   = ",".join(SCHEMA_COLUMNS)
    updates= ",".join(f"{c}=EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:])
    tmpl   = "(" + ",".join(f"%({c})s" for c in SCHEMA_COLUMNS) + ")"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO btc_weekly ({cols}) VALUES %s "
            f"ON CONFLICT (week_start) DO UPDATE SET {updates}",
            [row],
            template=tmpl,
        )
    conn.commit()

# ─── Fetch gold from Yahoo ───────────────────────────────────────────────────
async def _fetch_yahoo_gold(start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Gold: {start.date()} → {end.date()}")
    try:
        raw = await asyncio.to_thread(
            yf.download,
            "GC=F",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        debug(f"  raw.shape = {raw.shape}")
        if raw.empty:
            warn("  yahoo returned no rows for GC=F")
            return pd.DataFrame()
        # drop ticker-level MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(0)
        price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if price_col not in raw.columns:
            warn(f"  missing column '{price_col}' in raw: {list(raw.columns)}")
            return pd.DataFrame()
        series = raw[price_col].copy()
        series.name = "gold_price"
        series.index = pd.to_datetime(series.index).tz_localize("UTC")
        debug(f"  series.shape = {series.shape}")
        return series.to_frame()
    except Exception as e:
        warn(f"  gold fetch error: {e}")
        return pd.DataFrame()

# ─── Fetch BTC via CoinGecko (fallback to Yahoo) ─────────────────────────────
async def _fetch_coingecko(client: httpx.AsyncClient, start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"BTC (CG): {start.date()} → {end.date()}")
    try:
        params = {"vs_currency":"usd","days":(end-start).days+1,"interval":"daily"}
        r = await client.get(COINGECKO_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("prices", [])
        df = pd.DataFrame(data, columns=["ts","price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("D")
        df = df.set_index("date")[["price"]].rename(columns={"price":"close_usd"})
        debug(f"  CG df.shape = {df.shape}")
        return df
    except Exception as e:
        warn(f"  CG fetch error: {e}")

    # Fallback to Yahoo
    debug("  Fallback: BTC via Yahoo")
    raw = await asyncio.to_thread(
        yf.download,
        "BTC-USD",
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    debug(f"  raw.shape = {raw.shape}")
    if raw.empty:
        warn("  yahoo BTC returned no rows")
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    if col not in raw.columns:
        warn(f"  missing column '{col}' in raw: {list(raw.columns)}")
        return pd.DataFrame()
    s = raw[col].copy()
    s.name = "close_usd"
    s.index = pd.to_datetime(s.index).tz_localize("UTC")
    df = s.to_frame()
    debug(f"  series.shape = {df.shape}")
    return df

# ─── Fetch CoinMetrics ────────────────────────────────────────────────────────
async def _fetch_coinmetrics(client: httpx.AsyncClient, start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"CoinMetrics: {start.date()} → {end.date()}")
    params = {
        "assets":"btc","metrics":"CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency":"1d","start_time":start.strftime("%Y-%m-%d"),
        "end_time":end.strftime("%Y-%m-%d"),
    }
    r = await client.get(COINMETRICS_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        warn("  CoinMetrics returned no data")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("date")
    for c in ["CapRealUSD","SplyCur","CapMrktCurUSD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["realised_price"] = df["CapRealUSD"] / df["SplyCur"]
    df["nupl"]           = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
    debug(f"  CM df.shape = {df.shape}")
    return df[["realised_price","nupl"]]

# ─── Fetch FRED series ────────────────────────────────────────────────────────
async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"FRED {series_id}: {start.date()} → {end.date()}")
    try:
        r = await client.get(FRED_URL.format(series_id=series_id), timeout=30)
        r.raise_for_status()
        txt = r.text
        df = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
        df.index = df.index.tz_localize("UTC")
        col = FRED_MAP.get(series_id, series_id.lower())
        df.columns = [col]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[[col]].loc[start:end]
        debug(f"  FRED df.shape = {df.shape}")
        return df
    except Exception as e:
        warn(f"  FRED {series_id} error: {e}")
        return pd.DataFrame()

# ─── Main ingestion ──────────────────────────────────────────────────────────
async def ingest_weekly(week_anchor=None, years: int = 1):
    debug(f"INGEST start: anchor={week_anchor}, years={years}")
    # Normalize anchor
    if isinstance(week_anchor, datetime) and week_anchor.tzinfo is None:
        week_anchor = week_anchor.replace(tzinfo=timezone.utc)
    now = week_anchor or datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    debug(f"  Window = {start.date()} to {now.date()}")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = {
            "btc":     _fetch_coingecko(client, start, now),
            "cm":      _fetch_coinmetrics(client, start, now),
            "fed_liq": _fetch_fred_series(client, "WALCL", start, now),
            "ecb_liq": _fetch_fred_series(client, "ECBASSETS", start, now),
            "dxy":     _fetch_fred_series(client, "DTWEXBGS", start, now),
            "ust10":   _fetch_fred_series(client, "DGS10", start, now),
            "gold":    _fetch_yahoo_gold(start, now),
            "spx":     _fetch_fred_series(client, "SP500", start, now),
        }
        debug(f"  Launching tasks: {list(tasks.keys())}")
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        dfs = {}
        for key, res in zip(tasks.keys(), results):
            if isinstance(res, Exception):
                warn(f"  Task {key} failed: {res}")
                dfs[key] = pd.DataFrame()
            else:
                dfs[key] = res
                debug(f"  {key} -> {res.shape}")

    # Merge, fill, and clean
    df_all = pd.concat([df for df in dfs.values() if not df.empty], axis=1).sort_index().ffill()
    debug(f"  Merged shape: {df_all.shape}")
    df_all.dropna(subset=["close_usd"], inplace=True)
    debug(f"  After drop shape: {df_all.shape}")
    if df_all.empty:
        warn("  No data to upsert, aborting.")
        return

    weekly = df_all.resample("W-MON", label="left", closed="left").last()
    debug(f"  Weekly shape: {weekly.shape}")

    # Upsert into DB
    conn = get_db_connection()
    _create_table_if_missing(conn)
    count = 0
    for ts, row in weekly.iterrows():
        rec = {c: None for c in SCHEMA_COLUMNS}
        rec["week_start"] = ts
        for k, v in row.items():
            rec[k] = None if pd.isna(v) else float(v)
        _upsert_row(conn, rec)
        count += 1

    debug(f"  Upserted {count} rows")

# ─── CLI Entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=1))
