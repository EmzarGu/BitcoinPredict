#!/usr/bin/env python3
from __future__ import annotations

# Restore built-in str
import builtins
str = builtins.str

import asyncio, os
from datetime import datetime, timedelta, timezone
import httpx, pandas as pd, psycopg2, psycopg2.extras, yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Print-based diagnostics

def debug(msg: str): print(f"[DEBUG] {msg}")
def warn(msg: str): print(f"[WARN] {msg}")

# Schema
SCHEMA_COLUMNS = [
    "week_start", "close_usd", "realised_price", "nupl",
    "fed_liq", "ecb_liq", "dxy", "ust10",
    "gold_price", "spx_index",
]

# URLs
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_MAP = {"WALCL":"fed_liq","ECBASSETS":"ecb_liq","DTWEXBGS":"dxy","DGS10":"ust10","SP500":"spx_index"}

# DB

def get_db_connection():
    url = os.getenv("DATABASE_URL")
    if not url: raise RuntimeError("DATABASE_URL missing")
    return psycopg2.connect(url)

def _create_table_if_missing(conn):
    sql="""
    CREATE TABLE IF NOT EXISTS btc_weekly (
      week_start TIMESTAMPTZ PRIMARY KEY,
      close_usd REAL, realised_price REAL, nupl REAL,
      fed_liq REAL, ecb_liq REAL, dxy REAL, ust10 REAL,
      gold_price REAL, spx_index REAL
    );
    """
    cur=conn.cursor(); cur.execute(sql); conn.commit()

def _upsert_row(conn,row):
    cols=",".join(SCHEMA_COLUMNS)
    upd=",".join(f"{c}=EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:])
    tmpl="("+".join(f"%({c})s" for c in SCHEMA_COLUMNS)+")"
    psycopg2.extras.execute_values(
        conn.cursor(),
        f"INSERT INTO btc_weekly ({cols}) VALUES %s ON CONFLICT (week_start) DO UPDATE SET {upd}",
        [row],template=tmpl
    )
    conn.commit()

# Fetchers
async def _fetch_yahoo_gold(start: datetime, end: datetime) -> pd.DataFrame:
    debug(f"Gold: {start} to {end}")
    try:
        raw=await asyncio.to_thread(yf.download,'GC=F',start=start.strftime('%Y-%m-%d'),end=(end+timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=True,progress=False)
        debug(f"Raw gold shape: {raw.shape}")
        if raw.empty: warn("No gold rows");return pd.DataFrame()
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.droplevel(0)
        col='Adj Close' if 'Adj Close' in raw.columns else 'Close'
        if col not in raw.columns: warn(f"No {col}");return pd.DataFrame()
        s=raw[col].copy(); s.name='gold_price'; s.index=pd.to_datetime(s.index).tz_localize('UTC')
        debug(f"Gold series {s.shape}")
        return s.to_frame()
    except Exception as e:
        warn(f"Gold error: {e}");return pd.DataFrame()

async def _fetch_coingecko(client,start,end):
    debug(f"CG: {start} to {end}")
    try:
        r=await client.get(COINGECKO_URL,params={'vs_currency':'usd','days':(end-start).days+1,'interval':'daily'})
        r.raise_for_status()
        df=pd.DataFrame(r.json().get('prices',[]),columns=['ts','price'])
        df['date']=pd.to_datetime(df.ts,unit='ms',utc=True).dt.floor('D')
        df=df.set_index('date')[['price']].rename(columns={'price':'close_usd'})
        debug(f"CG {df.shape}");return df
    except Exception as e:
        warn(f"CG err: {e}")
    debug("Fallback BTC")
    raw=await asyncio.to_thread(yf.download,'BTC-USD',start=start.strftime('%Y-%m-%d'),end=(end+timedelta(days=1)).strftime('%Y-%m-%d'),interval='1d',auto_adjust=False,progress=False)
    if raw.empty: warn("No BTC rows");return pd.DataFrame()
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.droplevel(1)
    col='Adj Close' if 'Adj Close' in raw.columns else 'Close'
    s=raw[col].copy();s.name='close_usd';s.index=pd.to_datetime(s.index).tz_localize('UTC')
    debug(f"BTC series {s.shape}");return s.to_frame()

async def _fetch_coinmetrics(client,start,end):
    debug(f"CM: {start} to {end}")
    r=await client.get(COINMETRICS_URL,params={'assets':'btc','metrics':'CapRealUSD,SplyCur,CapMrktCurUSD','frequency':'1d','start_time':start.strftime('%Y-%m-%d'),'end_time':end.strftime('%Y-%m-%d')});r.raise_for_status()
    data=r.json().get('data',[])
    if not data: warn("No CM data");return pd.DataFrame()
    df=pd.DataFrame(data);df['date']=pd.to_datetime(df.time,utc=True);df=df.set_index('date')
    for c in['CapRealUSD','SplyCur','CapMrktCurUSD']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df['realised_price']=df.CapRealUSD/df.SplyCur;df['nupl']=(df.CapMrktCurUSD-df.CapRealUSD)/df.CapMrktCurUSD
    debug(f"CM {df.shape}");return df[['realised_price','nupl']]

async def _fetch_fred_series(client,series,start,end):
    debug(f"FRED {series}: {start} to {end}")
    try:
        r=await client.get(FRED_URL.format(series_id=series));r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text),index_col=0,parse_dates=True);df.index=df.index.tz_localize('UTC')
        col=FRED_MAP.get(series,series.lower());df.columns=[col];df[col]=pd.to_numeric(df[col],errors='coerce')
        df=df[[col]].loc[start:end]
        debug(f"FRED {df.shape}");return df
    except Exception as e:
        warn(f"FRED {series} err: {e}");return pd.DataFrame()

# Ingest
async def ingest_weekly(anchor=None,years=1):
    debug(f"INGEST anchor={anchor},years={years}")
    if isinstance(anchor,datetime) and anchor.tzinfo is None:anchor=anchor.replace(tzinfo=timezone.utc)
    now=anchor or datetime.now(timezone.utc);start=now-timedelta(days=365*years)
    debug(f"Window {start} to {now}")
    async with httpx.AsyncClient() as client:
        tasks={'btc':_fetch_coingecko(client,start,now),'cm':_fetch_coinmetrics(client,start,now),'fed_liq':_fetch_fred_series(client,'WALCL',start,now),'ecb_liq':_fetch_fred_series(client,'ECBASSETS',start,now),'dxy':_fetch_fred_series(client,'DTWEXBGS',start,now),'ust10':_fetch_fred_series(client,'DGS10',start,now),'gold':_fetch_yahoo_gold(start,now),'spx':_fetch_fred_series(client,'SP500',start,now)}
        debug(f"Tasks {list(tasks)}")
        results=await asyncio.gather(*tasks.values(),return_exceptions=True)
        dfs={k:(pd.DataFrame() if isinstance(r,Exception) else r) for k,r in zip(tasks,results)}
        for k,df in dfs.items():debug(f"{k} {df.shape}")
    df_all=pd.concat([df for df in dfs.values() if not df.empty],axis=1).sort_index().ffill()
    debug(f"Merged {df_all.shape}")
    df_all.dropna(subset=['close_usd'],inplace=True)
    debug(f"Cleaned {df_all.shape}")
    if df_all.empty:warn("No data");return
    weekly=df_all.resample('W-MON',label='left',closed='left').last()
    debug(f"Weekly {weekly.shape}")
    conn=get_db_connection();_create_table_if_missing(conn)
    ctr=0
    for ts,row in weekly.iterrows():rec={'week_start':ts};rec.update({c:(None if pd.isna(v) else float(v)) for c,v in row.items()});[rec.setdefault(c,None) for c in SCHEMA_COLUMNS];_upsert_row(conn,rec);ctr+=1
    debug(f"Upserted {ctr}")

# CLI
if __name__=='__main__':asyncio.run(ingest_weekly(datetime.now(timezone.utc),years=1))
