#!/usr/bin/env python3
from __future__ import annotations

# Restore built-in str to avoid shadowing issues
import builtins
str = builtins.str

import asyncio
import os
from datetime import datetime, timedelta, timezone, date as _date, time as _time
from typing import Any, Dict, List

import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf
from dotenv import load_dotenv

# Configuration
load_dotenv()

# Simple debug functions
def debug(msg: str): print(f"DEBUG: {msg}")
def warn(msg: str): print(f"WARNING: {msg}")

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

# Utility

def to_python_float(v: Any) -> float | None:
    return None if pd.isna(v) or v is None else float(v)


def get_db_connection():
    url = os.getenv("DATABASE_URL")
    if not url: raise RuntimeError("DATABASE_URL missing in .env")
    return psycopg2.connect(url)


def _create_table_if_missing(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS btc_weekly (
        week_start TIMESTAMPTZ PRIMARY KEY,
        close_usd REAL, realised_price REAL, nupl REAL,
        fed_liq REAL, ecb_liq REAL, dxy REAL, ust10 REAL,
        gold_price REAL, spx_index REAL
    );
    """
    with conn.cursor() as cur: cur.execute(sql)
    conn.commit()


def _upsert_row(conn, row: Dict[str, Any]):
    cols = ",".join(SCHEMA_COLUMNS)
    updates = ",".join(f"{c}=EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:])
    tmpl = "(" + ",".join(f"%({c})s" for c in SCHEMA_COLUMNS) + ")"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO btc_weekly ({cols}) VALUES %s ON CONFLICT (week_start) DO UPDATE SET {updates}",
            [row], template=tmpl
        )
    conn.commit()

# Fetch functions
async def _fetch_yahoo_gold(start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Fetching gold GC=F {start} to {end}")
    try:
        raw = await asyncio.to_thread(
            yf.download,
            "GC=F",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False
        )
        debug(f"Raw shape: {raw.shape}")
        if raw.empty:
            warn("Yahoo gold empty")
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(0)
        price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if price_col not in raw.columns:
            warn(f"Missing col {price_col}")
            return pd.DataFrame()
        s = raw[price_col].copy()
        s.name = "gold_price"
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        debug(f"Series shape: {s.shape}")
        return s.to_frame()
    except Exception as e:
        warn(f"Gold fetch error: {e}")
        return pd.DataFrame()

async def _fetch_coingecko(client: httpx.AsyncClient, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        params = {"vs_currency": "usd", "days": (end-start).days+1, "interval":"daily"}
        r = await client.get(COINGECKO_URL, params=params, timeout=30); r.raise_for_status()
        df = pd.DataFrame(r.json().get("prices",[]), columns=["ts","price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("D")
        df = df.set_index("date")[['price']].rename(columns={'price':'close_usd'})
        return df
    except Exception as e:
        warn(f"Coingecko fail {e}");
    # fallback Yahoo
    try:
        raw = await asyncio.to_thread(
            yf.download,"BTC-USD",
            start=start.strftime("%Y-%m-%d"), end=(end+timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=False, progress=False
        )
        if raw.empty: return pd.DataFrame()
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.droplevel(1)
        col='Adj Close' if 'Adj Close' in raw.columns else 'Close'
        s=raw[col].copy(); s.name='close_usd'; s.index=pd.to_datetime(s.index,utc=True)
        return s.to_frame()
    except Exception as e:
        warn(f"Yahoo BTC fail {e}")
    return pd.DataFrame()

async def _fetch_coinmetrics(client: httpx.AsyncClient, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    params={"assets":"btc","metrics":"CapRealUSD,SplyCur,CapMrktCurUSD","frequency":"1d",
            "start_time":start_date.strftime("%Y-%m-%d"),"end_time":end_date.strftime("%Y-%m-%d")}
    r = await client.get(COINMETRICS_URL, params=params, timeout=30); r.raise_for_status()
    dat=r.json().get("data",[])
    if not dat: return pd.DataFrame()
    df=pd.DataFrame(dat); df['date']=pd.to_datetime(df['time'],utc=True); df=df.set_index('date')
    for c in ['CapRealUSD','SplyCur','CapMrktCurUSD']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['realised_price']=df['CapRealUSD']/df['SplyCur']; df['nupl']=(df['CapMrktCurUSD']-df['CapRealUSD'])/df['CapMrktCurUSD']
    return df[['realised_price','nupl']]

async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        url=FRED_URL.format(series_id=series_id)
        r=await client.get(url,timeout=30); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text),index_col=0,parse_dates=True)
        df.index=df.index.tz_localize('UTC'); col=FRED_COLUMN_MAP.get(series_id,series_id.lower())
        df.columns=[col]; df[col]=pd.to_numeric(df[col],errors='coerce')
        return df[[col]].loc[start:end]
    except Exception as e:
        warn(f"FRED {series_id} fail {e}")
        return pd.DataFrame()

# Main ingestion
async def ingest_weekly(week_anchor=None, years=1):
    debug(f"ingest start: {week_anchor}, {years}")
    if isinstance(week_anchor, datetime) and week_anchor.tzinfo is None:
        week_anchor=week_anchor.replace(tzinfo=timezone.utc)
    elif isinstance(week_anchor,_date) and not isinstance(week_anchor,datetime):
        week_anchor=datetime.combine(week_anchor,_time(),tzinfo=timezone.utc)
    now=week_anchor or datetime.now(timezone.utc)
    start=now - timedelta(days=365*years)
    debug(f"window {start} to {now}")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks={
            'btc': _fetch_coingecko(client,start,now),
            'cm': _fetch_coinmetrics(client,start,now),
            'fed_liq': _fetch_fred_series(client,'WALCL',start,now),
            'ecb_liq': _fetch_fred_series(client,'ECBASSETS',start,now),
            'dxy': _fetch_fred_series(client,'DTWEXBGS',start,now),
            'ust10': _fetch_fred_series(client,'DGS10',start,now),
            'gold': _fetch_yahoo_gold(start,now),
            'spx': _fetch_fred_series(client,'SP500',start,now),
        }
        debug(f"tasks {list(tasks)}")
        results=await asyncio.gather(*tasks.values(),return_exceptions=True)
        dfs={}
        for k,r in zip(tasks,results):
            if isinstance(r,Exception): warn(f"task {k} error {r}"); dfs[k]=pd.DataFrame()
            else: dfs[k]=r; debug(f"{k} {r.shape}")
    df_all=pd.concat([df for df in dfs.values() if not df.empty],axis=1).sort_index().ffill()
    debug(f"combined {df_all.shape}")
    df_all.dropna(subset=['close_usd'],inplace=True)
    debug(f"post drop {df_all.shape}")
    if df_all.empty: warn("no data"); return
    weekly=df_all.resample('W-MON',label='left',closed='left').last()
    debug(f"weekly {weekly.shape}")
    rows=[]
    for ts,row in weekly.iterrows(): rows.append({**{'week_start':ts},**{k:to_python_float(v) for k,v in row.items()}})
    conn=get_db_connection(); _create_table_if_missing(conn)
    for r in rows: _upsert_row(conn,r)
    debug(f"upserted {len(rows)}")

# CLI
if __name__=='__main__':
    asyncio.run(ingest_weekly(datetime.now(timezone.utc),years=1))
