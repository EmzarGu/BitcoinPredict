import os
import argparse
import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import httpx
import asyncio
import io

# This line loads your DATABASE_URL and any API keys from your .env file
load_dotenv()

# --- Database Connection (Using the working DATABASE_URL method) ---
def get_db_connection():
    """Establishes a connection using the DATABASE_URL from the .env file."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment. Please check your .env file.")
    return psycopg2.connect(database_url)

def create_table_if_not_exists(conn):
    """Creates the btc_weekly table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS btc_weekly (
                week_start TIMESTAMPTZ PRIMARY KEY,
                close_usd REAL,
                realised_price REAL,
                nupl REAL,
                fed_liq REAL,
                ecb_liq REAL,
                dxy REAL,
                ust10 REAL,
                gold_price REAL,
                spx_index REAL
            );
        """)
        conn.commit()

# --- Helper Function to Fix Data Type Bug ---
def to_python_float(value):
    """Converts a pandas/numpy number to a standard Python float, or None."""
    if pd.isna(value) or value is None:
        return None
    return float(value)

# --- Restored Data Fetching Logic ---
async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, col_name: str) -> pd.DataFrame:
    """Fetches a single data series from FRED."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        df.columns = [col_name]
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        return df
    except Exception as e:
        print(f"Failed to fetch FRED series {series_id}: {e}")
    return pd.DataFrame()

async def _fetch_coinmetrics(client: httpx.AsyncClient, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetches on-chain metrics from CoinMetrics."""
    params = {
        "assets": "btc",
        "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency": "1d",
        "start_time": start_date.strftime("%Y-%m-%d"),
        "end_time": end_date.strftime("%Y-%m-%d"),
    }
    try:
        resp = await client.get("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('date')
        for col in ["CapRealUSD", "SplyCur", "CapMrktCurUSD"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["realised_price"] = df["CapRealUSD"] / df["SplyCur"]
        df["nupl"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
        return df[["realised_price", "nupl"]]
    except Exception as e:
        print(f"Failed to fetch CoinMetrics data: {e}")
    return pd.DataFrame()

def _get_yfinance_data(ticker, start_date, end_date, column_name):
    """Synchronous version for yfinance calls inside asyncio."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = [column_name]
            return df
    except Exception as e:
        print(f"An error occurred fetching {ticker} from Yahoo Finance: {e}")
    return pd.DataFrame()

# --- Main Ingestion Function (Now an async function) ---
async def ingest_weekly(week_anchor, years=1):
    """Main async function to ingest weekly data and upsert to database."""
    end_date = week_anchor
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    async with httpx.AsyncClient() as client:
        loop = asyncio.get_running_loop()
        # Create all fetching tasks
        tasks = {
            "cm": _fetch_coinmetrics(client, start_date, end_date),
            "fed_liq": _fetch_fred_series(client, "WALCL", "fed_liq"),
            "ecb_liq": _fetch_fred_series(client, "ECBASSETS", "ecb_liq"),
            "btc": loop.run_in_executor(None, _get_yfinance_data, 'BTC-USD', start_date, end_date, 'close_usd'),
            "gold": loop.run_in_executor(None, _get_yfinance_data, 'GC=F', start_date, end_date, 'gold_price'),
            "spx": loop.run_in_executor(None, _get_yfinance_data, '^GSPC', start_date, end_date, 'spx_index'),
            "dxy": loop.run_in_executor(None, _get_yfinance_data, 'DX-Y.NYB', start_date, end_date, 'dxy'),
            "ust10": loop.run_in_executor(None, _get_yfinance_data, '^TNX', start_date, end_date, 'ust10'),
        }
        results = await asyncio.gather(*tasks.values())
        dataframes = dict(zip(tasks.keys(), results))

    if dataframes["btc"].empty:
        print("❌ Critical error: Could not fetch Bitcoin data. Aborting.")
        return

    # Merge all dataframes
    merged_df = pd.concat([df for df in dataframes.values() if not df.empty], axis=1)
    merged_df.ffill(inplace=True)
    merged_df.dropna(subset=['close_usd'], inplace=True)

    if merged_df.empty:
        print("No data to process after merging. Aborting.")
        return

    weekly_
