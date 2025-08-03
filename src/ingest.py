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

# --- Restored Data Fetching Logic from Original Script ---
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
        if not data:
            return pd.DataFrame()
            
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

# --- Main Ingestion Function ---
async def _ingest_async(week_anchor, years=1):
    end_date = week_anchor
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    # Using an async client for FRED and CoinMetrics
    async with httpx.AsyncClient() as client:
        # Create tasks for all concurrent fetches
        cm_task = _fetch_coinmetrics(client, start_date, end_date)
        fed_liq_task = _fetch_fred_series(client, "WALCL", "fed_liq")
        ecb_liq_task = _fetch_fred_series(client, "ECBASSETS", "ecb_liq")
        
        # Run yfinance fetches in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        btc_df = await loop.run_in_executor(None, _get_yfinance_data, 'BTC-USD', start_date, end_date, 'close_usd')
        gold_df = await loop.run_in_executor(None, _get_yfinance_data, 'GC=F', start_date, end_date, 'gold_price')
        spx_df = await loop.run_in_executor(None, _get_yfinance_data, '^GSPC', start_date, end_date, 'spx_index')
        dxy_df = await loop.run_in_executor(None, _get_yfinance_data, 'DX-Y.NYB', start_date, end_date, 'dxy')
        ust10_df = await loop.run_in_executor(None, _get_yfinance_data, '^TNX', start_date, end_date, 'ust10')
        
        cm_df, fed_liq_df, ecb_liq_df = await asyncio.gather(cm_task, fed_liq_task, ecb_liq_task)

    if btc_df.empty:
        print("❌ Critical error: Could not fetch Bitcoin data. Aborting.")
        return

    # Merge all dataframes
    all_dfs = [btc_df, cm_df, fed_liq_df, ecb_liq_df, gold_df, spx_df, dxy_df, ust10_df]
    # Use reduce for cleaner merging of multiple frames
    merged_df = pd.concat([df for df in all_dfs if not df.empty], axis=1)

    merged_df.ffill(inplace=True)
    merged_df.dropna(subset=['close_usd'], inplace=True) # Only drop if core data is missing

    if merged_df.empty:
        print("No data to process after merging. Aborting.")
        return

    weekly_df = merged_df.resample('W-MON').last()

    # --- Upsert to Database ---
    print("Connecting to database...")
    try:
        with get_db_connection() as conn:
            print("✅ Database connection successful.")
            create_table_if_not_exists(conn)
            with conn.cursor() as cur:
                data_to_upsert = [
                    (
                        idx,
                        to_python_float(row.get('close_usd')),
                        to_python_float(row.get('realised_price')),
                        to_python_float(row.get('nupl')),
                        to_python_float(row.get('fed_liq')),
                        to_python_float(row.get('ecb_liq')),
                        to_python_float(row.get('dxy')),
                        to_python_float(row.get('ust10')),
                        to_python_float(row.get('gold_price')),
                        to_python_float(row.get('spx_index'))
                    )
                    for idx, row in weekly_df.iterrows()
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO btc_weekly (week_start, close_usd, realised_price, nupl, fed_liq, ecb_liq, dxy, ust10, gold_price, spx_index)
                    VALUES %s
                    ON CONFLICT (week_start) DO UPDATE SET
                        close_usd = EXCLUDED.close_usd, realised_price = EXCLUDED.realised_price, nupl = EXCLUDED.nupl,
                        fed_liq = EXCLUDED.fed_liq, ecb_liq = EXCLUDED.ecb_liq, dxy = EXCLUDED.dxy,
                        ust10 = EXCLUDED.ust10, gold_price = EXCLUDED.gold_price, spx_index = EXCLUDED.spx_index;
                    """,
                    data_to_upsert
                )
            conn.commit()
        print(f"✅ Successfully ingested and upserted {len(weekly_df)} weeks of data.")
    except Exception as e:
        print(f"❌ An error occurred during the database operation: {e}")

def ingest_weekly(week_anchor, years=1):
    """Public wrapper to run the async ingestion logic."""
    asyncio.run(_ingest_async(week_anchor, years))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest historical market data into the database.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    args = parser.parse_args()

    ingest_weekly(datetime.now(timezone.utc), years=args.years)
