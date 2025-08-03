import os
import argparse
import pandas as pd
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values

# --- Database Connection (Using the standard environment variable method) ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    # This function assumes that environment variables are correctly set
    # in the execution environment (e.g., your notebook or shell).
    return psycopg2.connect(
        host=os.environ.get('DB_HOST'),
        port=os.environ.get('DB_PORT', '5432'), # Default to 5432 if not set
        database=os.environ.get('DB_NAME'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD')
    )

def create_table_if_not_exists(conn):
    """Creates the btc_weekly table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS btc_weekly (
                week_start TIMESTAMP PRIMARY KEY,
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

# --- Data Ingestion Logic (Corrected) ---
def get_yfinance_data(ticker, start_date, end_date, column_name):
    """Generic function to fetch data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = [column_name]
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred with Yahoo Finance for {ticker}: {e}")
        return pd.DataFrame()

def ingest_weekly(week_anchor, years=1):
    """Ingests weekly data and upserts it into the database."""
    end_date = week_anchor
    start_date = end_date - timedelta(days=365 * years)

    # --- Fetch Data ---
    # (Data fetching logic is correct)
    # ...

    # --- Upsert to Database ---
    print("Attempting to connect to the database...")
    try:
        with get_db_connection() as conn:
            print("✅ Database connection successful.")
            create_table_if_not_exists(conn)
            # (Rest of the database logic)
            # ...
    except psycopg2.OperationalError as e:
        print("❌ DATABASE CONNECTION FAILED.")
        print("This is an ENVIRONMENT issue, not a code issue.")
        print("Please check your DB_HOST, DB_USER, etc. environment variables.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# (The rest of the file remains the same)
