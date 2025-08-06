import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
import argparse
from datetime import datetime

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from src.features import build_features
from sklearn.model_selection import train_test_split

def generate_forecast(forecast_date: str = None):
    """
    Generates a full, unified, and logically consistent 4-week and 12-week forecast.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if forecast_date:
        target_date = pd.to_datetime(forecast_date, utc=True)
        time_diffs = np.abs(features_df.index - target_date)
        latest_features = features_df.iloc[[time_diffs.argmin()]]
    else:
        training_features = build_features(for_training=True)
        latest_features = training_features.tail(1)

    classifier = joblib.load(models_dir / "direction_classifier.joblib")
    price_target_4w_model = joblib.load(models_dir / "price_target_4w.joblib")
    price_target_12w_model = joblib.load(models_dir / "price_target_12w.joblib")
    
    training_cols = classifier.get_booster().feature_names
    X_latest = latest_features[training_cols]
    
    # --- 2. Calculate Historical Volatility ---
    # We use the historical daily returns to get a realistic measure of volatility.
    daily_returns = features_df['close_usd'].pct_change()
    # Calculate the annualized volatility and then scale it to our forecast horizons
    annual_volatility = daily_returns.std() * np.sqrt(365)
    volatility_4w = annual_volatility / np.sqrt(52/4) # Scale to 4 weeks
    volatility_12w = annual_volatility / np.sqrt(52/12) # Scale to 12 weeks

    # --- 3. Implement Liquidity Regime Filter ---
    features_df['Liquidity_Z'] = -features_df['dxy_z']
    features_df['DXY_26w_MA'] = features_df['dxy'].rolling(window=26).mean()
    features_df['DXY_26w_trend'] = features_df['dxy'] / features_df['DXY_26w_MA']
    features_df['Risk-On'] = np.where((features_df['Liquidity_Z'] > 0) & (features_df['DXY_26w_trend'] < 1), True, False)
    
    regime_status = "Risk-On" if features_df.loc[latest_features.index, 'Risk-On'].iloc[0] else "Risk-Off"
    
    # --- 4. Generate the "Smart" Ensemble Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest)[0]
    direction = ["Bearish", "Neutral", "Bullish"][np.argmax(direction_probabilities)]

    last_close_price = X_latest['close_usd'].iloc[0]

    # 4-Week Forecast
    return_4w = price_target_4w_model.predict(X_latest)[0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    # Create the range based on the target and adjusted by probabilities
    lower_bound_4w = price_target_4w * (1 - volatility_4w * (1 + direction_probabilities[0] - direction_probabilities[2]))
    upper_bound_4w = price_target_4w * (1 + volatility_4w * (1 - direction_probabilities[0] + direction_probabilities[2]))

    # 12-Week Forecast
    return_12w = price_target_12w_model.predict(X_latest)[0]
    price_target_12w = last_close_price * (1 + return_12w)

    lower_bound_12w = price_target_12w * (1 - volatility_12w * (1 + direction_probabilities[0] - direction_probabilities[2]))
    upper_bound_12w = price_target_12w * (1 + volatility_12w * (1 - direction_probabilities[0] + direction_probabilities[2]))
    
    # --- 5. Assemble and Print Final Forecast ---
    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {latest_features.index[0].strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    print(f"Directional Outlook: {direction} (Probabilities: Bearish {direction_probabilities[0]:.2%}, Neutral {direction_probabilities[1]:.2%}, Bullish {direction_probabilities[2]:.2%})")
    print("\n--- 4-Week Outlook ---")
    print(f"Price Target: ${price_target_4w:,.2f}")
    print(f"Likely Range: ${lower_bound_4w:,.2f} - ${upper_bound_4w:,.2f}")
    print("\n--- 12-Week Outlook ---")
    print(f"Price Target: ${price_target_12w:,.2f}")
    print(f"Likely Range: ${lower_bound_12w:,.2f} - ${upper_bound_12w:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
