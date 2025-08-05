import argparse
from datetime import datetime
import sys
from pathlib import Path
import asyncio # Import the asyncio library

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import ingest

# --- THIS IS THE FIX (Part 1) ---
# Make the backfill function asynchronous so it can 'await' other async functions
async def backfill(years):
    """
    Backfills historical data for the specified number of years.
    
    :param years: Number of years to backfill.
    """
    print(f"Starting backfill for {years} year(s)...")
    today = datetime.now()
    
    # --- THIS IS THE FIX (Part 2) ---
    # Use 'await' to correctly run the async ingest_weekly function
    await ingest.ingest_weekly(today, years=years)
    print("Backfill complete.")

def main():
    parser = argparse.ArgumentParser(description='Backfill historical market data.')
    parser.add_argument('--years', type=int, required=True, help='Number of years to backfill.')
    args = parser.parse_args()
    
    # Run the main async function using asyncio.run()
    asyncio.run(backfill(args.years))

if __name__ == "__main__":
    main()
