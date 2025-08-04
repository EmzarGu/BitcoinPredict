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

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_API_KEY = os.getenv("FRED_API_KEY")

# This mapping is from your original script
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

def _init_db(conn: psycopg2.extensions.connection, row: Dict[str, Any]) -> None:
    """Ensure table exists and upsert a single row."""
    _create_table_if_missing(conn)
    columns = ",".join(SCHEMA_COLUMNS)
    update = ",".join([f"{c} = EXCLUDED.{c}" for c in SCHEMA_COLUMNS[1:]])
    with conn.cursor() as cur:
        template = "(" + ",".join([f"%({col})s" for col in SCHEMA_COLUMNS]) + ")"
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO btc_weekly ({columns}) VALUES %s ON CONFLICT (week_start) DO UPDATE SET {update}",
            [row],
            template=template,
        )
    conn.commit()


# --- All Data Fetching Functions (Restored to your original, robust logic) ---

async def _fetch_yahoo_gold(start, end) -> pd.DataFrame:
    """Specific fetcher for Gold from Yahoo as a fallback."""
    try:
        # This is the verified function from our successful notebook test
        raw = await asyncio.to_thread(yf.download, "GC=F", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("Yahoo Finance returned no data for gold.")
            return pd.DataFrame()

        price_col_tuple = ('Close', 'GC=F')
        if price_col_tuple not in raw.columns:
            logger.warning(f"Required column '{price_col_tuple}' not found in Yahoo Finance gold data.")
            return pd.DataFrame()

        df = raw[[price_col_tuple]].copy()
        df.columns = ["gold_price"]
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        logger.warning(f"An error occurred in _fetch_yahoo_gold: {e}")
    return pd.DataFrame()


async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, start, end) -> pd.DataFrame:
    """Fetches a single data series from FRED, with fallback for Gold."""
    url = FRED_URL.format(series_id=series_id)
    column_name = FRED_COLUMN_MAP.get(series_id, series_id.lower())
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        df.index = df.index.tz_localize('UTC')
        df.columns = [column_name]
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

        if series_id == "GOLDAMGBD228NLBM" and (df.empty or df[column_name].isna().all()):
            logger.warning("FRED data for gold is unavailable. Falling back to Yahoo Finance.")
            return await _fetch_yahoo_gold(start, end)
            
        return df[[column_name]]
    except Exception as e:
        logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        if series_id == "GOLDAMGBD228NLBM":
            logger.warning("Falling back to Yahoo Finance for gold price.")
            return await _fetch_yahoo_gold(start, end)
    return pd.DataFrame()


async def _fetch_yahoo_btc(start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """Fetches BTC price from Yahoo, restored from your original script."""
    try:
        kwargs = {"interval": "1d", "auto_adjust": False}
        if start and end:
            kwargs.update({"start": start, "end": end})
        else:
            kwargs.update({"period": "1mo"})

        raw = await asyncio.to_thread(yf.download, "BTC-USD", **kwargs)
        if raw.empty: return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        
        price_col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
        if price_col not in raw.columns: return pd.DataFrame()

        series = raw[price_col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        
        series.name = "close_usd"
        series.index = pd.to_datetime(series.index, utc=True)
        df = series.to_frame()
        df["volume"] = pd.NA
        return df[["close_usd", "volume"]]
    except Exception as e:
        logger.warning(f"Failed to fetch BTC from Yahoo Finance: {e}")
    return pd.DataFrame()


async def _fetch_coingecko(client: httpx.AsyncClient, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """Fetches Bitcoin prices from CoinGecko, with Yahoo Finance as fallback."""
    try:
        if start and end: # Prioritize Yahoo for specific ranges as in original
            return await _fetch_yahoo_btc(start, end)

        # Your original CoinGecko logic with retries and fallback
        params = {"vs_currency": "usd", "days": 8, "interval": "daily"}
        resp = await client.get(COINGECKO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prices = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms", utc=True).dt.floor("D")
        df = prices.set_index("date")[["price"]].rename(columns={"price": "close_usd"})
        df["volume"] = pd.NA # Placeholder as per original
        return df

    except Exception as e:
        logger.warning(f"CoinGecko failed ({e}), falling back to Yahoo Finance.")
        return await _fetch_yahoo_btc(start, end)


async def _fetch_coinmetrics(client: httpx.AsyncClient, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetches on-chain metrics from CoinMetrics."""
    params = {"assets": "btc", "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD", "frequency": "1d", "start_time": start_date.strftime("%Y-%m-%d"), "end_time": end_date.strftime("%Y-%m-%d")}
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


# --- Main Ingestion Logic (Based on your original) ---
async def ingest_weekly(week_anchor=None, years=1):
    """Main async function to ingest weekly data and upsert to database."""
    now = week_anchor or datetime.now(timezone.utc)
    week_start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = now
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # This task setup is from your original script
        tasks = {
            "btc": _fetch_coingecko(client, start=start_date, end=end_date),
            "cm": _fetch_coinmetrics(client, start_date=start_date, end_date=end_date),
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
    if "volume" in merged_df.columns:
        merged_df = merged_df.drop(columns=["volume"])
    
    merged_df = merged_df.sort_index().ffill()
    merged_df.dropna(subset=['close_usd'], inplace=True)

    if merged_df.empty:
        print("No data to process after merging. Aborting.")
        return
    
    # This logic correctly processes all weeks of data for backfill
    weekly_df = merged_df.resample('W-MON', label="left", closed="left").last()
    
    data_to_upsert = []
    for idx, row in weekly_df.iterrows():
        if idx.date() < start_date.date(): continue # Skip data outside the requested range
        final_row = row.to_dict()
        final_row['week_start'] = idx
        for key, value in final_row.items():
            if key != 'week_start':
                final_row[key] = to_python_float(value)
        for col in SCHEMA_COLUMNS:
            if col not in final_row:
                final_row[col] = None
        data_to_upsert.append(final_row)

    if not data_to_upsert:
        print("⚠️ No new weekly data to insert.")
        return

    print("Connecting to database...")
    try:
        with get_db_connection() as conn:
            print("✅ Database connection successful.")
            _create_table_if_missing(conn)
            # Loop to insert each row individually using your original _init_db logic
            for row_data in data_to_upsert:
                _init_db(conn, row_data)
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
