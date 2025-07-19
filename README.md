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
