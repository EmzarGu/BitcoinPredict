import asyncio
import io
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import argparse

import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError
from dotenv import load_dotenv

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

FRED_COLUMN_MAP = {
    "WALCL": "fed_liq", "ECBASSETS": "ecb_liq", "DTWEXBGS": "dxy",
    "DGS10": "ust10", "GOLDAMGBD228NLBM": "gold_price", "SP500": "spx_index",
}

def to_python_float(value):
    if pd.isna(value) or value is None:
        return None
    return float(value)

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found. Please check your .env file.")
    return psycopg2.connect(database_url)

def _create_table_if_missing(conn: psycopg2.extensions.connection) -> None:
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

async def _fetch_yahoo_gold(start, end) -> pd.DataFrame:
    try:
        raw = await asyncio.to_thread(yf.download, "GC=F", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("Yahoo Finance returned no data for gold.")
            return pd.DataFrame()

        price_col_tuple = ('Close', 'GC=F')
        if price_col_tuple not in raw.columns:
            logger.warning(f"Required column '{price_col_tuple}' not found in Yahoo Finance gold data. Available columns: {raw.columns.tolist()}")
            return pd.DataFrame()

        df = raw[[price_col_tuple]].copy()
        df.columns = ["gold_price"]
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        logger.warning(f"An error occurred in _fetch_yahoo_gold: {e}")
    return pd.DataFrame()

async def _fetch_fred_series(client: httpx.AsyncClient, series_id: str, start, end) -> pd.DataFrame:
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
    try:
        if start and end:
            return await _fetch_yahoo_btc(start, end)

        params = {"vs_currency": "usd", "days": 8, "interval": "daily"}
        resp = await client.get(COINGECKO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prices = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms", utc=True).dt.floor("D")
        df = prices.set_index("date")[["price"]].rename(columns={"price": "close_usd"})
        df["volume"] = pd.NA
        return df

    except Exception as e:
        logger.warning(f"CoinGecko failed ({e}), falling back to Yahoo Finance.")
        return await _fetch_yahoo_btc(start, end)

async def _fetch_coinmetrics(client: httpx.AsyncClient, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    print("--- DEBUG: Attempting to fetch from CoinMetrics ---")
    params = {
        "assets": "btc",
        "metrics": "CapRealUSD,SplyCur,CapMrktCurUSD",
        "frequency": "1d",
        "start_time": start_date.strftime("%Y-%m-%d"),
        "end_time": end_date.strftime("%Y-%m-%d")
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = await client.get(COINMETRICS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                print(f"--- DEBUG: CoinMetrics returned no data on attempt {attempt + 1} ---")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
                else:
                    print("--- DEBUG: All CoinMetrics attempts failed, returning empty DataFrame. ---")
                    return pd.DataFrame()

            print("--- DEBUG: CoinMetrics data received successfully. ---")
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['time'], utc=True)
            df = df.set_index('date')
            
            for col in ["CapRealUSD", "SplyCur", "CapMrktCurUSD"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df["realised_price"] = df["CapRealUSD"] / df["SplyCur"]
            df["nupl"] = (df["CapMrktCurUSD"] - df["CapRealUSD"]) / df["CapMrktCurUSD"]
            
            return df[["realised_price", "nupl"]]

        except Exception as e:
            print(f"--- DEBUG: Error fetching CoinMetrics on attempt {attempt + 1}: {e} ---")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
            else:
                print("--- DEBUG: All retry attempts for CoinMetrics failed. ---")
                return pd.DataFrame()

async def ingest_weekly(week_anchor=None, years=1):
    now = week_anchor or datetime.now(timezone.utc)
    week_start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = now
    start_date = end_date - timedelta(days=365 * years)

    print("Fetching market data...")
    async with httpx.AsyncClient(follow_redirects=True) as client:
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

    print("\n--- DEBUG: Inspecting dataframes before merge ---")
    if not dataframes["gold"].empty:
        print("Gold dataframe is NOT empty. Last 5 rows:")
        print(dataframes["gold"].tail())
    else:
        print("Gold dataframe IS EMPTY.")
    if not dataframes["cm"].empty:
        print("CoinMetrics dataframe is NOT empty. Last 5 rows:")
        print(dataframes["cm"].tail())
    else:
        print("CoinMetrics dataframe IS EMPTY.")

    merged_df = pd.concat([df for df in dataframes.values() if not df.empty], axis=1)
    if "volume" in merged_df.columns:
        merged_df = merged_df.drop(columns=["volume"])
    
    print("\n--- DEBUG: After pd.concat, before ffill ---")
    print("Last 5 rows of merged_df:")
    print(merged_df.tail())
    if 'realised_price' in merged_df.columns:
        print("Realised prices in last 5 rows:")
        print(merged_df['realised_price'].tail())
    else:
        print("Realised price column MISSING after concat.")
    
    merged_df = merged_df.sort_index().ffill()
    
    print("\n--- DEBUG: After ffill ---")
    print("Last 5 rows of merged_df:")
    print(merged_df.tail())
    if 'realised_price' in merged_df.columns:
        print("Realised prices in last 5 rows:")
        print(merged_df['realised_price'].tail())
    else:
        print("Realised price column MISSING after ffill.")

    merged_df.dropna(subset=['close_usd'], inplace=True)

    if merged_df.empty:
        print("No data to process after merging. Aborting.")
        return
    
    weekly_df = merged_df.resample('W-MON', label="left", closed="left").last()
    
    data_to_upsert = []
    for idx, row in weekly_df.iterrows():
        if idx.date() < start_date.date(): continue
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
            for row_data in data_to_upsert:
                _init_db(conn, row_data)
        print(f"✅ Successfully ingested and upserted {len(data_to_upsert)} weeks of data.")
    except Exception as e:
        print(f"❌ An error occurred during the database operation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest historical market data.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    parser.add_argument('--date', type=str, default=None, help='Anchor date for the ingestion in YYYY-MM-DD format. Defaults to today.')
    args = parser.parse_args()

    anchor_date = None
    if args.date:
        try:
            anchor_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("❌ Error: Date format must be YYYY-MM-DD.")
            sys.exit(1)
    
    asyncio.run(ingest_weekly(week_anchor=anchor_date, years=args.years))
