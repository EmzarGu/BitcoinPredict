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

## FRED API access
Some economic series fetched from FRED may not be available via the simple CSV endpoint used by the ingestor.
For those series you can provide a personal FRED API key through the `FRED_API_KEY` environment variable:

```bash
export FRED_API_KEY="your_fred_key"
```

When a requested series cannot be retrieved (for example if the key is missing or the API returns an error), the script continues and the affected column will be filled with `NA` values.

If the gold price series (`GOLDAMGBD228NLBM`) is unavailable from FRED, the ingestor automatically falls back to downloading daily gold prices from [Stooq](https://stooq.com) and resamples them to weekly values.
