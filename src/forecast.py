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
    Generates a full, unified forecast using the final ensemble of models,
    now led by the Logistic Regression classifier.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Could not build features for forecasting. Aborting.")
        return

    try:
        # Load the new Logistic Regression classifier and the final scaler
        classifier = joblib.load(models_dir / "direction_classifier_logreg.joblib")
        price_reg_4w = joblib.load(models_dir / "price_target_4w_bayes.joblib")
        price_reg_12w = joblib.load(models_dir / "price_target_12w_bayes.joblib")
        scaler = joblib.load(models_dir / "scaler_final.joblib")
    except Exception as e:
        print(f"❌ A model or scaler could not be loaded: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    if forecast_date:
        target_date = pd.to_datetime(forecast_date, utc=True)
        # Find the single row of features for the specified date
        try:
            latest_features = features_df.loc[[target_date]]
        except KeyError:
            print(f"❌ Forecast date {forecast_date} not found in the dataset.")
            return
    else:
        # Get the most recent features available
        latest_features = features_df.tail(1)

    last_close_price = latest_features['close_usd'].iloc[0]

    # --- 3. Prepare Data for Prediction ---
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in latest_features.columns]
    X_latest = latest_features[predictor_cols_exist]
    
    # Scale the features using the loaded scaler
    X_latest_scaled = scaler.transform(X_latest)

    # --- 4. Generate the Ensemble Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest_scaled)[0]

    # Implement Liquidity Regime Filter
    regime_status = "Risk-Off"
    if 'Liquidity_Z' in X_latest.columns and 'DXY_26w_trend' in X_latest.columns:
        if not X_latest.empty and X_latest['Liquidity_Z'].iloc[0] > 0 and X_latest['DXY_26w_trend'].iloc[0] < 1:
            regime_status = "Risk-On"

    if regime_status == "Risk-Off":
        bullish_prob = direction_probabilities[2]
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction * 0.5
        direction_probabilities[0] += reduction * 0.5

    direction = ["Bearish", "Neutral", "Bullish"][np.argmax(direction_probabilities)]

    # 4-Week Forecast
    return_4w = price_reg_4w.predict(X_latest_scaled)[0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    # 12-Week Forecast
    return_12w = price_reg_12w.predict(X_latest_scaled)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Assemble and Print Final Forecast ---
    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {latest_features.index[0].strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    print(f"Directional Outlook: {direction} (Probabilities: Bearish {direction_probabilities[0]:.2%}, Neutral {direction_probabilities[1]:.2%}, Bullish {direction_probabilities[2]:.2%})")
    print("\n--- 4-Week Outlook ---")
    print(f"Price Target: ${price_target_4w:,.2f}")
    print("\n--- 12-Week Outlook ---")
    print(f"Price Target: ${price_target_12w:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
