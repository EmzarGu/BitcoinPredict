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

# --- Add the new LGC_distance feature ---
FEATURE_COLS: List[str] = [
    "Momentum_4w", "Momentum_12w", "Momentum_26w", "Realised_Price_Delta",
    "nupl", "dxy_z", "ust10_z", "gold_price_z", "spx_index_z",
    "DXY_Invert", "LGC_distance", "Target", "Target_12w"
]

def _load_btc_weekly() -> pd.DataFrame:
    # ... (function is unchanged)
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            conn = psycopg2.connect(db_url)
            query = (
                "SELECT week_start, close_usd, realised_price, nupl, "
                "dxy, ust10, gold_price, spx_index "
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
        logger.error("No data source found for btc_weekly")
        return pd.DataFrame()

def build_features(lookback_weeks: int = 260, for_training: bool = True) -> pd.DataFrame:
    df = _load_btc_weekly()
    if df.empty:
        return df

    df = df.set_index("week_start")
    df = df.sort_index()

    # --- THIS IS THE NEW LGC CALCULATION ---
    # Ensure there are no zero or negative prices before taking the log
    df_for_lgc = df[df['close_usd'] > 0].copy()
    
    def log_growth_curve(x, a, b):
        return a + b * np.log(x)

    x_data = np.arange(1, len(df_for_lgc) + 1)
    y_data = np.log(df_for_lgc['close_usd'])

    try:
        params, _ = curve_fit(log_growth_curve, x_data, y_data)
        df['lgc'] = np.exp(log_growth_curve(np.arange(1, len(df) + 1), *params))
        df['LGC_distance'] = (df['close_usd'] / df['lgc']) - 1
    except Exception as e:
        logger.warning(f"Could not fit LGC: {e}")
        df['LGC_distance'] = 0 # Default to zero if the curve fails
    # ----------------------------------------
    
    df["Momentum_4w"] = df["close_usd"].pct_change(4)
    df["Momentum_12w"] = df["close_usd"].pct_change(12)
    df["Momentum_26w"] = df["close_usd"].pct_change(26)

    df["Realised_Price_Delta"] = df["close_usd"] / df["realised_price"] - 1

    for col in ["dxy", "ust10", "gold_price", "spx_index"]:
        rolling = df[col].rolling(window=52)
        df[f"{col}_z"] = (df[col] - rolling.mean()) / rolling.std()
    
    df["DXY_Invert"] = 1 / df["dxy"]

    df["Target"] = df["close_usd"].shift(-4) / df["close_usd"] - 1
    df["Target_12w"] = df["close_usd"].shift(-12) / df["close_usd"] - 1

    if for_training:
        predictor_cols = [col for col in FEATURE_COLS if "Target" not in col]
        df = df.dropna(subset=predictor_cols)
        df = df.tail(lookback_weeks)
    
    df = df.sort_index()
    return df
