import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


FEATURE_COLS: List[str] = [
    "Momentum_4w",
    "Momentum_12w",
    "Momentum_26w",
    "Realised_Price_Delta",
    "nupl",
    "fed_liq_z",
    "ecb_liq_z",
    "dxy_z",
    "ust10_z",
    "gold_price_z",
    "spx_index_z",
    "Combined_Liq",
    "Liq_Change",
    "DXY_Invert",
    "Target",
]


def _load_btc_weekly() -> pd.DataFrame:
    """Load btc_weekly data from the database or CSV fallback."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            conn = psycopg2.connect(db_url)
            query = (
                "SELECT week_start, close_usd, realised_price, nupl, "
                "fed_liq, ecb_liq, dxy, ust10, gold_price, spx_index "
                "FROM btc_weekly ORDER BY week_start"
            )
            df = pd.read_sql(query, conn)
            conn.close()
            df["week_start"] = pd.to_datetime(df["week_start"], utc=True)
            df = df.sort_values("week_start").reset_index(drop=True)
            return df
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Failed to read database: %s", exc)
    try:
        df = pd.read_csv("data/btc_weekly_latest.csv")
        df["week_start"] = pd.to_datetime(df["week_start"], utc=True)
        df = df.sort_values("week_start").reset_index(drop=True)
        logger.info("Loaded %s rows from CSV fallback", len(df))
        return df
    except FileNotFoundError:
        logger.error("No data source found for btc_weekly")
        return pd.DataFrame(columns=[
            "week_start",
            "close_usd",
            "realised_price",
            "nupl",
            "fed_liq",
            "ecb_liq",
            "dxy",
            "ust10",
            "gold_price",
            "spx_index",
        ])


def build_features(lookback_weeks: int = 260) -> pd.DataFrame:
    """Build feature dataframe from btc_weekly raw data."""
    df = _load_btc_weekly()
    if df.empty:
        return df

    df = df.set_index("week_start")
    df = df.sort_index()

    df["Momentum_4w"] = df["close_usd"].pct_change(4)
    df["Momentum_12w"] = df["close_usd"].pct_change(12)
    df["Momentum_26w"] = df["close_usd"].pct_change(26)

    df["Realised_Price_Delta"] = df["close_usd"] / df["realised_price"] - 1

    for col in ["fed_liq", "ecb_liq", "dxy", "ust10", "gold_price", "spx_index"]:
        rolling = df[col].rolling(window=52)
        df[f"{col}_z"] = (df[col] - rolling.mean()) / rolling.std()

    df["Combined_Liq"] = df["fed_liq"] + df["ecb_liq"]
    df["Liq_Change"] = df["Combined_Liq"].pct_change(52)
    df["DXY_Invert"] = 1 / df["dxy"]

    df["Target"] = df["close_usd"].shift(-4) / df["close_usd"] - 1

    df = df.tail(lookback_weeks + 52 + 4)  # ensure lookback window

    df = df.dropna(subset=FEATURE_COLS)

    df = df.tail(lookback_weeks)

    df = df.sort_index()
    return df


def save_latest_features(df: pd.DataFrame, path: str = "artifacts/features_latest.parquet") -> None:
    """Save the most recent feature row to a parquet file."""
    if df.empty:
        logger.warning("No features to save")
        return
    latest = df.tail(1)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    latest.to_parquet(dest)
    logger.info("Saved latest features to %s", dest)


if __name__ == "__main__":
    features = build_features()
    save_latest_features(features)
    print(features.tail(3))
