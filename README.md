# BitcoinPredict

BitcoinPredict provides utilities to ingest weekly Bitcoin market data and related macroeconomic indicators. The collected data can be stored in a PostgreSQL/TimescaleDB instance for further analysis and forecasting.

## Prerequisites
- Python 3.12 or later
- (Optional) PostgreSQL/TimescaleDB database if you want to store ingested data

## Installation
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running tests
Execute the test suite with:

```bash
pytest -q
```

## Database configuration
Set the `DATABASE_URL` environment variable to enable writing to your
PostgreSQL/TimescaleDB instance. The value should be a standard PostgreSQL
connection string. For example:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/bitcoin"
```

If this variable is not set, the ingestion script will skip the database upsert
step.

The ingestion utility will create the `btc_weekly` table automatically if it is
missing. When the TimescaleDB extension is available, the table is converted
into a hypertable, but it also works with a plain PostgreSQL database.

## FRED API access
Some economic series fetched from FRED may not be available via the simple CSV endpoint used by the ingestor.
For those series you can provide a personal FRED API key through the `FRED_API_KEY` environment variable:

```bash
export FRED_API_KEY="your_fred_key"
```

When a requested series cannot be retrieved (for example if the key is missing or the API returns an error), the script continues and the affected column will be filled with `NA` values.

Gold prices are sourced directly from Yahoo Finance using the GLD ETF ticker and resampled to weekly values.

## Coingecko API access
Some endpoints on Coingecko now require an API key. If you encounter authorization errors, set the `COINGECKO_API_KEY` environment variable:

```bash
export COINGECKO_API_KEY="your_coingecko_key"
```

When Coingecko requests fail, the ingestor falls back to fetching prices from Yahoo Finance.
