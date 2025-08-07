import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import psycopg2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PREDICTOR_COLS: List[str] = [
    "SMA_ratio_52w",
    "LGC_distance_z",
    "Liquidity_Z",
    "Nupl_Z",
    "Realised_to_Spot",
    "RSI_14w",
    "DXY_26w_trend",
    "gold_corr_26w",
    "spx_corr_26w",
]

def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
        except Exception as exc:
            logger.warning("Failed to read database: %s", exc)

    try:
        df = pd.read_csv("data/btc_weekly_latest.csv")
        df["week_start"] = pd.to_datetime(df["week_start"], utc=True)
        df = df.sort_values("week_start").reset_index(drop=True)
        logger.info("Loaded %s rows from CSV fallback", len(df))
        return df
    except FileNotFoundError:
        logger.error("No data source found for btc_weekly. Please run ingest.py.")
        return pd.DataFrame()

def build_features(lookback_weeks: int = 260, for_training: bool = True) -> pd.DataFrame:
    df = _load_btc_weekly()
    if df.empty:
        return df

    df = df.set_index("week_start")
    df = df.sort_index()
    df['close_usd'] = pd.to_numeric(df['close_usd'], errors='coerce')

    # --- Feature Engineering ---
    df["SMA_ratio_52w"] = df["close_usd"] / df["close_usd"].rolling(window=52).mean()

    df_for_lgc = df[df['close_usd'] > 0].copy()
    def log_growth_curve(x, a, b): return a + b * np.log(x)
    x_data = np.arange(1, len(df_for_lgc) + 1)
    y_data = np.log(df_for_lgc['close_usd'])
    try:
        params, _ = curve_fit(log_growth_curve, x_data, y_data)
        df['lgc'] = np.exp(log_growth_curve(np.arange(1, len(df) + 1), *params))
        df['LGC_distance'] = (df['close_usd'] / df['lgc']) - 1
        rolling_lgc = df['LGC_distance'].rolling(window=52)
        df['LGC_distance_z'] = (df['LGC_distance'] - rolling_lgc.mean()) / rolling_lgc.std()
    except Exception as e:
        logger.warning(f"Could not fit LGC: {e}")
        df['LGC_distance_z'] = 0

    df['global_liq'] = df['fed_liq'] + df['ecb_liq']
    rolling_liq = df['global_liq'].pct_change(periods=52).rolling(window=52)
    df['Liquidity_Z'] = (df['global_liq'].pct_change(periods=52) - rolling_liq.mean()) / rolling_liq.std()

    rolling_nupl = df['nupl'].rolling(window=52)
    df['Nupl_Z'] = (df['nupl'] - rolling_nupl.mean()) / rolling_nupl.std()

    df["Realised_to_Spot"] = df["close_usd"] / df["realised_price"]
    df["RSI_14w"] = _calculate_rsi(df["close_usd"])
    df["DXY_26w_trend"] = df['dxy'] / df['dxy'].rolling(window=26).mean()

    btc_returns = df['close_usd'].pct_change()
    gold_returns = df['gold_price'].pct_change()
    spx_returns = df['spx_index'].pct_change()
    df['gold_corr_26w'] = btc_returns.rolling(window=26).corr(gold_returns)
    df['spx_corr_26w'] = btc_returns.rolling(window=26).corr(spx_returns)

    # --- Target Variables ---
    df["Target"] = df["close_usd"].shift(-4) / df["close_usd"] - 1
    df["Target_12w"] = df["close_usd"].shift(-12) / df["close_usd"] - 1

    # --- Final Processing ---
    final_cols = ['close_usd', 'Target', 'Target_12w'] + PREDICTOR_COLS
    existing_cols = [col for col in final_cols if col in df.columns]
    df = df[existing_cols]

    # **THIS IS THE FIX**: Apply NaN dropping conditionally.
    if for_training:
        # For training, we need complete rows with no NaNs in any column.
        df.dropna(inplace=True)
        df = df.tail(lookback_weeks)
    else:
        # For forecasting, we only need the predictors to be non-NaN.
        # This keeps the most recent rows where only Targets are NaN.
        df.dropna(subset=PREDICTOR_COLS, inplace=True)

    df = df.sort_index()
    return df
