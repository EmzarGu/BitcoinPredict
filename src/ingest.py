import os
import argparse
import pandas as pd
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values

# --- Database Connection (Reverted to the original, working method) ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using environment variables."""
    return psycopg2.connect(
        host=os.environ.get('DB_HOST'),
        port=os.environ.get('DB_PORT'),
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

# --- Data Ingestion ---

def get_btc_price_coingecko(start_date, end_date):
    """Fetches Bitcoin price data from CoinGecko."""
    cg = CoinGeckoAPI()
    try:
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        btc_data = cg.get_coin_market_chart_range_by_id(
            id='bitcoin', vs_currency='usd',
            from_timestamp=start_ts, to_timestamp=end_ts
        )
        prices = {datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d'): p for ts, p in btc_data['prices']}
        df = pd.DataFrame(prices.items(), columns=['date', 'close_usd'])
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    except Exception as e:
        print(f"An error occurred with CoinGecko: {e}")
        return pd.DataFrame()

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
    btc_df = get_btc_price_coingecko(start_date, end_date)
    if btc_df.empty:
        print("Falling back to Yahoo Finance for Bitcoin data.")
        btc_df = get_yfinance_data('BTC-USD', start_date, end_date, 'close_usd')

    gold_df = get_yfinance_data('GC=F', start_date, end_date, 'gold_price')
    spx_df = get_yfinance_data('^GSPC', start_date, end_date, 'spx_index')
    dxy_df = get_yfinance_data('DX-Y.NYB', start_date, end_date, 'dxy')
    ust10_df = get_yfinance_data('^TNX', start_date, end_date, 'ust10')
    
    # --- Merge Data ---
    merged_df = btc_df
    for df in [gold_df, spx_df, dxy_df, ust10_df]:
        if not df.empty:
            merged_df = merged_df.merge(df, how='left', left_index=True, right_index=True)

    # Use ffill() to fix deprecation warning
    merged_df.ffill(inplace=True) 
    merged_df.dropna(inplace=True)
    
    weekly_df = merged_df.resample('W').last()
    
    # --- Upsert to Database ---
    print("Connecting to the database...")
    with get_db_connection() as conn:
        print("Database connection successful.")
        create_table_if_not_exists(conn)
        with conn.cursor() as cur:
            data_to_upsert = [
                (idx, row.get('close_usd'), None, None, None, None, row.get('dxy'), row.get('ust10'), row.get('gold_price'), row.get('spx_index'))
                for idx, row in weekly_df.iterrows()
            ]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest historical market data into the database.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    args = parser.parse_args()
    
    ingest_weekly(datetime.now(), years=args.years)
