import asyncio
import io
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf
from dotenv import load_dotenv

# This loads your DATABASE_URL and any API keys from your .env file
load_dotenv()

logger = logging.getLogger(__name__)

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

# --- All Data Fetching Functions (with the final bug fix) ---
async def _fetch_yahoo_data(ticker: str, start: datetime, end: datetime, col_name: str) -> pd.DataFrame:
    """Fetches data from Yahoo Finance."""
    try:
        data = await asyncio.to_thread(yf.download, ticker, start=start, end=end, auto_adjust=True, progress=False)
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = [col_name]
            return df
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker} from Yahoo Finance: {e}")
    return pd.DataFrame()

async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, col_name: str) -> pd.DataFrame:
    """Fetches a single data series from FRED."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        # FIX: This line adds the missing timezone information to prevent the TypeError
        df.index = df.index.tz_localize('UTC')
        df.columns = [col_name]
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
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
        logger.warning(f"Failed to fetch CoinMetrics data: {e}")
    return pd.DataFrame()

# --- Main Ingestion Logic ---
async def ingest_weekly(week_anchor, years=1):
    """Main async function to ingest weekly data and upsert to database."""
    end_date = week_anchor
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    async with httpx.AsyncClient() as client:
        tasks = {
            "btc": _fetch_yahoo_data('BTC-USD', start_date, end_date, 'close_usd'),
            "cm": _fetch_coinmetrics(client, start_date, end_date),
            "fed_liq": _fetch_fred_series(client, "WALCL", "fed_liq"),
            "ecb_liq": _fetch_fred_series(client, "ECBASSETS", "ecb_liq"),
            "dxy": _fetch_yahoo_data('DX-Y.NYB', start_date, end_date, 'dxy'),
            "ust10": _fetch_yahoo_data('^TNX', start_date, end_date, 'ust10'),
            "gold": _fetch_yahoo_data('GC=F', start_date, end_date, 'gold_price'),
            "spx": _fetch_yahoo_data('^GSPC', start_date, end_date, 'spx_index'),
        }
        results = await asyncio.gather(*tasks.values())
        dataframes = dict(zip(tasks.keys(), results))

    if dataframes["btc"].empty:
        print("❌ Critical error: Could not fetch Bitcoin data. Aborting.")
        return

    # Correctly merge all dataframes
    merged_df = pd.concat([df for df in dataframes.values() if not df.empty], axis=1)
    merged_df.ffill(inplace=True)
    merged_df.dropna(subset=['close_usd'], inplace=True) # Ensure core data exists

    if merged_df.empty:
        print("No data to process after merging. Aborting.")
        return

    weekly_df = merged_df.resample('W-MON', label="left", closed="left").last()

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
                
                if not data_to_upsert:
                    print("⚠️ No new weekly data to insert.")
                    return

                execute_values(
                    cur,
                    """
                    INSERT INTO btc_weekly (week_start, close_usd, realised_price, nupl, fed_liq, ecb_liq, dxy, ust10, gold_price, spx_index)
                    VALUES %s ON CONFLICT (week_start) DO UPDATE SET
                        close_usd = EXCLUDED.close_usd, realised_price = EXCLUDED.realised_price, nupl = EXCLUDED.nupl,
                        fed_liq = EXCLUDED.fed_liq, ecb_liq = EXCLUDED.ecb_liq, dxy = EXCLUDED.dxy,
                        ust10 = EXCLUDED.ust10, gold_price = EXCLUDED.gold_price, spx_index = EXCLUDED.spx_index;
                    """,
                    data_to_upsert
                )
            conn.commit()
        print(f"✅ Successfully ingested and upserted {len(data_to_upsert)} weeks of data.")
    except Exception as e:
        print(f"❌ An error occurred during the database operation: {e}")

# This block allows running the script from the command line, just like your original
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest historical market data into the database.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    args = parser.parse_args()
    asyncio.run(ingest_weekly(datetime.now(timezone.utc), years=args.years))
