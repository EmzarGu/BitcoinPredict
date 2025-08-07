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

from src.features import build_features, PREDICTOR_COLS

def generate_forecast(forecast_date: str = None):
    """
    Generates a full, unified forecast using the blueprint's ensemble logic.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Could not build features for forecasting. Aborting.")
        return

    try:
        classifier = joblib.load(models_dir / "direction_classifier_xgb.joblib")
        price_target_4w_model = joblib.load(models_dir / "price_target_4w_bayes.joblib")
        price_target_12w_model = joblib.load(models_dir / "price_target_12w_bayes.joblib")
    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    if forecast_date:
        target_date = pd.to_datetime(forecast_date, utc=True)
        time_diffs = np.abs(features_df.index - target_date)
        latest_features = features_df.iloc[[time_diffs.argmin()]]
    else:
        latest_features = features_df.tail(1)
        
    # Now this call will not fail, because 'close_usd' is in latest_features
    last_close_price = latest_features['close_usd'].iloc[0]

    # Use the imported PREDICTOR_COLS to get the features for prediction
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in latest_features.columns]
    X_latest = latest_features[predictor_cols_exist].fillna(0)

    # --- 3. Implement Liquidity Regime Filter ---
    regime_status = "Risk-Off"
    if 'Liquidity_Z' in X_latest.columns and 'DXY_26w_trend' in X_latest.columns:
        if not X_latest.empty and X_latest['Liquidity_Z'].iloc[0] > 0 and X_latest['DXY_26w_trend'].iloc[0] < 1:
            regime_status = "Risk-On"

    # --- 4. Generate the Ensemble Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest)[0]

    if regime_status == "Risk-Off":
        bullish_prob = direction_probabilities[2]
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction * 0.5
        direction_probabilities[0] += reduction * 0.5

    direction = ["Bearish", "Neutral", "Bullish"][np.argmax(direction_probabilities)]

    # 4-Week Forecast
    return_4w = price_target_4w_model.predict(X_latest)[0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    # 12-Week Forecast
    return_12w = price_target_12w_model.predict(X_latest)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Calculate Forecast Ranges ---
    X_4w = features_df.loc[features_df['Target'].dropna().index][predictor_cols_exist]
    errors_4w = features_df['Target'].dropna() - price_target_4w_model.predict(X_4w)
    
    X_12w = features_df.loc[features_df['Target_12w'].dropna().index][predictor_cols_exist]
    errors_12w = features_df['Target_12w'].dropna() - price_target_12w_model.predict(X_12w)

    range_modifier_4w = np.percentile(np.abs(errors_4w), 80)
    lower_bound_4w = price_target_4w * (1 - range_modifier_4w)
    upper_bound_4w = price_target_4w * (1 + range_modifier_4w)

    range_modifier_12w = np.percentile(np.abs(errors_12w), 80)
    lower_bound_12w = price_target_12w * (1 - range_modifier_12w)
    upper_bound_12w = price_target_12w * (1 + range_modifier_12w)

    # --- 6. Assemble and Print Final Forecast ---
    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {latest_features.index[0].strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    print(f"Directional Outlook: {direction} (Probabilities: Bearish {direction_probabilities[0]:.2%}, Neutral {direction_probabilities[1]:.2%}, Bullish {direction_probabilities[2]:.2%})")
    print("\n--- 4-Week Outlook ---")
    print(f"Price Target: ${price_target_4w:,.2f}")
    print(f"Likely Range (80% confidence): ${lower_bound_4w:,.2f} - ${upper_bound_4w:,.2f}")
    print("\n--- 12-Week Outlook ---")
    print(f"Price Target: ${price_target_12w:,.2f}")
    print(f"Likely Range (80% confidence): ${lower_bound_12w:,.2f} - ${upper_bound_12w:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
