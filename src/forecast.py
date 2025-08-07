import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
import argparse
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features, PREDICTOR_COLS
from src.train import create_sequences

def generate_forecast(forecast_date: str = None):
    """
    Generates a full, unified forecast with ranges and specific dates.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Could not build features. Aborting.")
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
        ref_date = pd.to_datetime(forecast_date, utc=True)
        try:
            end_loc = features_df.index.get_loc(ref_date)
            latest_features_for_lstm = features_df.iloc[max(0, end_loc - 3):end_loc + 1]
            latest_features_flat = features_df.loc[[ref_date]]
        except KeyError:
            print(f"❌ Forecast date {forecast_date} not found in the dataset.")
            return
    else:
        latest_features_for_lstm = features_df.tail(4)
        latest_features_flat = features_df.tail(1)
        ref_date = latest_features_flat.index[0]
        
    last_close_price = latest_features_flat['close_usd'].iloc[0]

    # --- 3. Prepare Data for Prediction ---
    X_latest_scaled_lstm = scaler.transform(latest_features_for_lstm[predictor_cols_exist])
    X_latest_seq = np.array([X_latest_scaled_lstm])
    
    X_latest_scaled_flat = scaler.transform(latest_features_flat[predictor_cols_exist])

    # --- 4. Generate Forecasts ---
    direction_probabilities = classifier.predict_proba(X_latest_scaled_flat)[0]

    return_4w = price_reg_4w.predict(X_latest_seq, verbose=0)[0][0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    return_12w = price_reg_12w.predict(X_latest_scaled_flat)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Calculate Forecast Ranges ---
    y_4w_full = features_df['Target'].dropna()
    X_4w_full_scaled = scaler.transform(features_df.loc[y_4w_full.index][predictor_cols_exist])
    X_seq_full, y_seq_full = create_sequences(pd.DataFrame(X_4w_full_scaled, index=y_4w_full.index), y_4w_full)
    preds_4w = price_reg_4w.predict(X_seq_full, verbose=0).flatten()
    errors_4w = y_seq_full - preds_4w
    range_mod_4w = np.percentile(np.abs(errors_4w), 80)
    
    y_12w_full = features_df['Target_12w'].dropna()
    X_12w_full_scaled = scaler.transform(features_df.loc[y_12w_full.index][predictor_cols_exist])
    preds_12w = price_reg_12w.predict(X_12w_full_scaled)
    errors_12w = y_12w_full - preds_12w
    range_mod_12w = np.percentile(np.abs(errors_12w), 80)

    # --- 6. Assemble and Print Final Forecast ---
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

    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {ref_date.strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    print(f"Directional Outlook: {direction} (Probabilities: Bearish {direction_probabilities[0]:.2%}, Neutral {direction_probabilities[1]:.2%}, Bullish {direction_probabilities[2]:.2%})")
    
    date_4w = ref_date + timedelta(weeks=4)
    print(f"\n--- 4-Week Outlook (for {date_4w.strftime('%Y-%m-%d')}) ---")
    print(f"Price Target: ${price_target_4w:,.2f}")
    print(f"Likely Range (80% confidence): ${price_target_4w * (1 - range_mod_4w):,.2f} - ${price_target_4w * (1 + range_mod_4w):,.2f}")

    date_12w = ref_date + timedelta(weeks=12)
    print(f"\n--- 12-Week Outlook (for {date_12w.strftime('%Y-%m-%d')}) ---")
    print(f"Price Target: ${price_target_12w:,.2f}")
    print(f"Likely Range (80% confidence): ${price_target_12w * (1 - range_mod_12w):,.2f} - ${price_target_12w * (1 + range_mod_12w):,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
