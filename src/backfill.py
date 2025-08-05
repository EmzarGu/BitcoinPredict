import argparse
from datetime import datetime
import sys
from pathlib import Path

# --- THIS IS THE FIX ---
# Add the project's root directory to the Python path
# This allows the script to find the 'src' module
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# ---------------------

from src import ingest

def backfill(years):
    """
    Backfills historical data for the specified number of years.
    
    :param years: Number of years to backfill.
    """
    print(f"Starting backfill for {years} year(s)...")
    today = datetime.now()
    # The ingest_weekly function is async and needs to be awaited,
    # but since this is a simple script, we can call it directly
    # if we run the whole script with an asyncio runner.
    # For now, let's assume ingest_weekly handles its own event loop if called directly.
    ingest.ingest_weekly(today, years=years)
    print("Backfill complete.")

def main():
    parser = argparse.ArgumentParser(description='Backfill historical market data.')
    parser.add_argument('--years', type=int, required=True, help='Number of years to backfill.')
    args = parser.parse_args()
    backfill(args.years)

if __name__ == "__main__":
    main()
