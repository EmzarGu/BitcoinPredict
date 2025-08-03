import os
import argparse
import pandas as pd
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

# Initialize CoinGecko API
cg = CoinGeckoAPI()

def get_btc_price_coingecko(start_date, end_date):
    """
    Fetches Bitcoin price data from CoinGecko.
    """
    try:
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        btc_data = cg.get_coin_market_chart_range_by_id(
            id='bitcoin',
            vs_currency='usd',
            from_timestamp=start_ts,
            to_timestamp=end_ts
        )
        
        prices = {datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d'): p for ts, p in btc_data['prices']}
        df = pd.DataFrame(prices.items(), columns=['date', 'close_usd'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df
    except Exception as e:
        print(f"An error occurred with CoinGecko: {e}")
        return pd.DataFrame()

def get_yfinance_data(ticker, start_date, end_date, column_name):
    """
    Generic function to fetch data from Yahoo Finance, ensuring a simple column structure.
    """
    try:
        # Download data, explicitly setting auto_adjust to handle warnings
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            # Select only the 'Close' column and create a new DataFrame
            df = data[['Close']].copy()
            # Set a simple, single-level column name
            df.columns = [column_name]
            
            # Validate data
            if 'price' in column_name or 'usd' in column_name:
                df = df[df[column_name] > 0]
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred with Yahoo Finance for {ticker}: {e}")
        return pd.DataFrame()

def main(years=1):
    """
    Main function to ingest data and save it to a CSV file.
    
    :param years: Number of years of historical data to fetch.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # Fetch data from sources
    btc_df = get_btc_price_coingecko(start_date, end_date)
    if btc_df.empty:
        print("Falling back to Yahoo Finance for Bitcoin data.")
        btc_df = get_yfinance_data('BTC-USD', start_date, end_date, 'close_usd')

    gold_df = get_yfinance_data('GC=F', start_date, end_date, 'gold_price')
    spx_df = get_yfinance_data('^GSPC', start_date, end_date, 'spx_index')
    dxy_df = get_yfinance_data('DX-Y.NYB', start_date, end_date, 'dxy')
    ust10_df = get_yfinance_data('^TNX', start_date, end_date, 'ust10')

    # --- Data Merging ---
    # Start with the Bitcoin data as the base
    merged_df = btc_df
    # List of dataframes to merge
    dataframes_to_merge = [gold_df, spx_df, dxy_df, ust10_df]

    for df in dataframes_to_merge:
        if not df.empty:
            merged_df = merged_df.merge(df, how='left', left_index=True, right_index=True)

    # Handle missing values
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.dropna(inplace=True)

    # --- Save to CSV ---
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_path = os.path.join(output_dir, f'btc_and_macro_data_{years}y.csv')
    merged_df.to_csv(file_path)
    
    print(f"Data ingestion complete. File saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest historical market data.')
    parser.add_argument('--years', type=int, default=1, help='Number of years of historical data to fetch.')
    args = parser.parse_args()
    main(years=args.years)
