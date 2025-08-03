import argparse
from datetime import datetime
from src import ingest

def backfill(years):
    """
    Backfills historical data for the specified number of years.
    
    :param years: Number of years to backfill.
    """
    print(f"Starting backfill for {years} year(s)...")
    today = datetime.now()
    ingest.ingest_weekly(today, years=years)
    print("Backfill complete.")

def main():
    parser = argparse.ArgumentParser(description='Backfill historical market data.')
    parser.add_argument('--years', type=int, required=True, help='Number of years to backfill.')
    args = parser.parse_args()
    backfill(args.years)

if __name__ == "__main__":
    main()
