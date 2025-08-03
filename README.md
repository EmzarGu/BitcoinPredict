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

Gold prices and macroeconomic series are sourced from Yahoo Finance and
resampled to weekly values.

## Coingecko API access
Some endpoints on Coingecko now require an API key. If you encounter authorization errors, set the `COINGECKO_API_KEY` environment variable:

```bash
export COINGECKO_API_KEY="your_coingecko_key"
```

When Coingecko requests fail, the ingestor falls back to fetching prices from Yahoo Finance.
