import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
import argparse
from datetime import datetime
from tensorflow.keras.models import load_model

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features, PREDICTOR_COLS

def generate_forecast(forecast_date: str = None):
    """
    Generates a full, unified forecast using the final hybrid model.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Could not build features for forecasting. Aborting.")
        return

    try:
        classifier = joblib.load(models_dir / "direction_classifier_final.joblib")
        price_reg_4w = load_model(models_dir / "price_target_4w_final.h5")
        price_reg_12w = joblib.load(models_dir / "price_target_12w_final.joblib")
        scaler = joblib.load(models_dir / "scaler_final.joblib")
    except Exception as e:
        print(f"❌ A model or scaler could not be loaded: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in features_df.columns]
    
    if forecast_date:
        target_date = pd.to_datetime(forecast_date, utc=True)
        try:
            end_loc = features_df.index.get_loc(target_date)
            latest_features_for_lstm = features_df.iloc[max(0, end_loc - 3):end_loc + 1]
            latest_features_flat = features_df.loc[[target_date]]
        except KeyError:
            print(f"❌ Forecast date {forecast_date} not found in the dataset.")
            return
    else:
        latest_features_for_lstm = features_df.tail(4)
        latest_features_flat = features_df.tail(1)
        
    last_close_price = latest_features_flat['close_usd'].iloc[0]

    # --- 3. Prepare Data for Prediction ---
    X_latest_scaled_lstm = scaler.transform(latest_features_for_lstm[predictor_cols_exist])
    X_latest_seq = np.array([X_latest_scaled_lstm])
    
    X_latest_scaled_flat = scaler.transform(latest_features_flat[predictor_cols_exist])

    # --- 4. Generate the Ensemble Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest_scaled_flat)[0]

    # Implement Liquidity Regime Filter
    regime_status = "Risk-Off"
    if 'Liquidity_Z' in latest_features_flat.columns and 'DXY_26w_trend' in latest_features_flat.columns:
        if not latest_features_flat.empty and latest_features_flat['Liquidity_Z'].iloc[0] > 0 and latest_features_flat['DXY_26w_trend'].iloc[0] < 1:
            regime_status = "Risk-On"

    if regime_status == "Risk-Off":
        bullish_prob = direction_probabilities[2]
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction * 0.5
        direction_probabilities[0] += reduction * 0.5

    direction = ["Bearish", "Neutral", "Bullish"][np.argmax(direction_probabilities)]

    # 4-Week Forecast (LSTM)
    return_4w = price_reg_4w.predict(X_latest_seq, verbose=0)[0][0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    # 12-Week Forecast (Bayesian)
    return_12w = price_reg_12w.predict(X_latest_scaled_flat)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Assemble and Print Final Forecast ---
    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {latest_features_flat.index[0].strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    print(f"Directional Outlook: {direction} (Probabilities: Bearish {direction_probabilities[0]:.2%}, Neutral {direction_probabilities[1]:.2%}, Bullish {direction_probabilities[2]:.2%})")
    print("\n--- 4-Week Outlook (LSTM) ---")
    print(f"Price Target: ${price_target_4w:,.2f}")
    print("\n--- 12-Week Outlook (Bayesian) ---")
    print(f"Price Target: ${price_target_12w:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
