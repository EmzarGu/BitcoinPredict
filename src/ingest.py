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
import psycopg2.extensions
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError
from dotenv import load_dotenv

# This loads your DATABASE_URL and any API keys from your .env file
load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS: List[str] = [
    "week_start", "close_usd", "realised_price", "nupl", "fed_liq",
    "ecb_liq", "dxy", "ust10", "gold_price", "spx_index",
]

COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_COLUMN_MAP = {
    "WALCL": "fed_liq", "ECBASSETS": "ecb_liq", "DTWEXBGS": "dxy",
    "DGS10": "ust10", "GOLDAMGBD228NLBM": "gold_price", "SP500": "spx_index",
}

# --- Helper Function to Fix the Database Data Type Bug ---
def to_python_float(value):
    """Converts a pandas/numpy number to a standard Python float, or None."""
    if pd.isna(value) or value is None:
        return None
    return float(value)

# --- Database Connection and Setup (from your original script) ---
def get_db_connection():
    """Establishes a connection using the DATABASE_URL from the .env file."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found. Please check your .env file.")
    return psycopg2.connect(database_url)

def _create_table_if_missing(conn: psycopg2.extensions.connection) -> None:
    """Create the ``btc_weekly`` table if it doesn't exist."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS btc_weekly (
        week_start TIMESTAMPTZ PRIMARY KEY,
        close_usd REAL, realised_price REAL, nupl REAL, fed_liq REAL,
        ecb_liq REAL, dxy REAL, ust10 REAL, gold_price REAL, spx_index REAL
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()

# --- All Data Fetching Functions (Restored to your original, robust logic) ---
async def _fetch_yahoo_data(ticker: str, start: datetime, end: datetime, col_name: str) -> pd.DataFrame:
    """Fetches data from Yahoo Finance using your original robust method."""
    try:
        raw = await asyncio.to_thread(yf.download, ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(0)
        
        price_col = 'Close'
        if 'Adj Close' in raw.columns:
            price_col = 'Adj Close'
        elif 'Close' not in raw.columns:
            logger.warning(f"Yahoo Finance data for {ticker} missing Close/Adj Close column")
            return pd.DataFrame()

        df = raw[[price_col]].copy()
        df.columns = [col_name]
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker} from Yahoo Finance: {e}")
    return pd.DataFrame()

async def _fetch_yahoo_gold(start: datetime, end: datetime) -> pd.DataFrame:
    """Specific fetcher for Gold from Yahoo as a fallback."""
    return await _fetch_yahoo_data("GC=F", start, end, "gold_price")

async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetches a single data series from FRED, with fallback for Gold."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    column_name = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        df.index = df.index.tz_localize('UTC') # FIX: Add timezone info
        df.columns = [column_name]
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df[[column_name]]
    except Exception as e:
        logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        # RESTORED: Your original fallback logic for gold price
        if series_id == "GOLDAMGBD228NLBM":
            logger.warning("Falling back to Yahoo Finance for gold price.")
            return await _fetch_yahoo_gold(start, end)
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
        resp = await client.get(COINMETRICS_URL, params=params, timeout=30)
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
        logger.warning(f"Failed to fetch CoinMetrics data: {e}")
    return pd.DataFrame()


# --- Main Ingestion Logic (Corrected for full backfill) ---
async def ingest_weekly(week_anchor=None, years=1):
    """Main async function to ingest weekly data and upsert to database."""
    now = week_anchor or datetime.now(timezone.utc)
    end_date = now
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = {
            "btc": _fetch_yahoo_data('BTC-USD', start_date, end_date, 'close_usd'),
            "cm": _fetch_coinmetrics(client, start_date, end_date),
            "fed_liq": _fetch_fred_series(client, "WALCL", start_date, end_date),
            "ecb_liq": _fetch_fred_series(client, "ECBASSETS", start_date, end_date),
            "dxy": _fetch_fred_series(client, "DTWEXBGS", start_date, end_date),
            "ust10": _fetch_fred_series(client, "DGS10", start_date, end_date),
            "gold": _fetch_fred_series(client, "GOLDAMGBD228NLBM", start_date, end_date),
            "spx": _fetch_fred_series(client, "SP500", start_date, end_date),
        }
        results = await asyncio.gather(*tasks.values())
        dataframes = dict(zip(tasks.keys(), results))

    if dataframes["btc"].empty:
        print("❌ Critical error: Could not fetch Bitcoin data. Aborting.")
        return

    # Correctly merge all dataframes
    merged_df = pd.concat([df for df in dataframes.values() if not df.empty], axis=1)
    merged_df.ffill(inplace=True)
    merged_df.dropna(subset=['close_usd'], inplace=True)

    if merged_df.empty:
        print("No data to process after merging. Aborting.")
        return

    # Resample to get weekly data
    weekly_df = merged_df.resample('W-MON', label="left", closed="left").last()
    
    # --- FIX: Prepare all weekly rows for insertion, not just one ---
    data_to_upsert = [
        {
            "week_start": idx,
            "close_usd": to_python_float(row.get('close_usd')),
            "realised_price": to_python_float(row.get('realised_price')),
            "nupl": to_python_float(row.get('nupl')),
            "fed_liq": to_python_float(row.get('fed_liq')),
            "ecb_liq": to_python_float(row.get('ecb_liq')),
            "dxy": to_python_float(row.get('dxy')),
            "ust10": to_python_float(row.get('ust10')),
            "gold_price": to_python_float(row.get('gold_price')),
            "spx_index": to_python_float(row.get('spx_index')),
        }
        for idx, row in weekly_df.iterrows()
    ]

    if not data_to_upsert:
        print("⚠️ No weekly data available to insert.")
        return

    print("Connecting to database...")
    try:
        with get_db_connection() as conn:
            print("✅ Database connection successful.")
            _create_table_if_missing(conn)
            
            # Use the original _init_db function in a loop for simplicity and correctness
            for row in data_to_upsert:
                with conn.cursor() as cur:
                     # This logic correctly prepares the data for a single row insert
                     columns = ",".join(row.keys())
                     update = ",".join([f"{c} = EXCLUDED.{c}" for c in row.keys() if c != 'week_start'])
                     template = f"({', '.join(['%s'] * len(row))})"
                     values = tuple(row.values())
                     
                     sql = f"INSERT INTO btc_weekly ({columns}) VALUES {template} ON CONFLICT (week_start) DO UPDATE SET {update};"
                     cur.execute(sql, values)
                conn.commit()

        print(f"✅ Successfully ingested and upserted {len(data_to_upsert)} weeks of data.")
    except Exception as e:
        print(f"❌ An error occurred during the database operation: {e}")


# This block allows running the script from the command line
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ingest historical market data.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    args = parser.parse_args()
    asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=args.years))
