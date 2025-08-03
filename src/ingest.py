import os
import argparse
import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datetime import datetime, timedelta

# This line loads your DATABASE_URL from your .env file, just like your original script.
load_dotenv()

# --- Database Connection (Using your original, working method) ---
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

# --- Data Ingestion (With corrected logic for data quality) ---
def get_yfinance_data(ticker, start_date, end_date, column_name):
    """Generic function to fetch data from Yahoo Finance."""
    try:
        # Fetch data and suppress the progress bar for cleaner logs
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = [column_name]
            return df
    except Exception as e:
        print(f"An error occurred fetching {ticker} from Yahoo Finance: {e}")
    return pd.DataFrame()

def ingest_weekly(week_anchor, years=1):
    """Ingests weekly data and upserts it into the database."""
    end_date = week_anchor
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    # --- Fetch Data ---
    btc_df = get_yfinance_data('BTC-USD', start_date, end_date, 'close_usd')
    gold_df = get_yfinance_data('GC=F', start_date, end_date, 'gold_price')
    spx_df = get_yfinance_data('^GSPC', start_date, end_date, 'spx_index')
    dxy_df = get_yfinance_data('DX-Y.NYB', start_date, end_date, 'dxy')
    ust10_df = get_yfinance_data('^TNX', start_date, end_date, 'ust10')

    if btc_df.empty:
        print("❌ Critical error: Could not fetch Bitcoin data. Aborting.")
        return

    # --- Merge Data ---
    merged_df = btc_df
    for df in [gold_df, spx_df, dxy_df, ust10_df]:
        if not df.empty:
            merged_df = merged_df.merge(df, how='left', left_index=True, right_index=True)

    # Use ffill() which is the modern replacement for the deprecated method
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True) # Drop any rows that still have NaNs (e.g., at the start)

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
                        row.get('close_usd'), None, None, None, None, # Placeholders for old columns
                        row.get('dxy'), row.get('ust10'),
                        row.get('gold_price'), row.get('spx_index')
                    )
                    for idx, row in weekly_df.iterrows()
                ]

                # This is the reliable upsert logic
                execute_values(
                    cur,
                    """
                    INSERT INTO btc_weekly (week_start, close_usd, realised_price, nupl, fed_liq, ecb_liq, dxy, ust10, gold_price, spx_index)
                    VALUES %s
                    ON CONFLICT (week_start) DO UPDATE SET
                        close_usd = EXCLUDED.close_usd, dxy = EXCLUDED.dxy, ust10 = EXCLUDED.ust10,
                        gold_price = EXCLUDED.gold_price, spx_index = EXCLUDED.spx_index;
                    """,
                    data_to_upsert
                )
            conn.commit()
        print(f"✅ Successfully ingested and upserted {len(weekly_df)} weeks of data.")
    except Exception as e:
        print(f"❌ An error occurred during the database operation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest historical market data into the database.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    args = parser.parse_args()

    ingest_weekly(datetime.now(), years=args.years)
