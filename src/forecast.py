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

from src.features import build_features
from src.train import create_sequences

def generate_forecast(forecast_date: str = None):
    """
    Generates a final, unified forecast using the complete hybrid model suite.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    try:
        classifier_4w = joblib.load(models_dir / "direction_classifier_4w_final.joblib")
        classifier_12w = joblib.load(models_dir / "direction_classifier_12w_final.joblib")
        price_reg_4w = load_model(models_dir / "price_target_4w_final.h5")
        price_reg_12w = joblib.load(models_dir / "price_target_12w_final.joblib")
        scaler_cls_4w = joblib.load(models_dir / "scaler_classifier_4w_final.joblib")
        scaler_cls_12w = joblib.load(models_dir / "scaler_classifier_12w_final.joblib")
        scaler_reg_4w = joblib.load(models_dir / "scaler_regressor_4w_final.joblib")
        scaler_reg_12w = joblib.load(models_dir / "scaler_regressor_12w_final.joblib")
    except Exception as e:
        print(f"❌ A model or scaler could not be loaded: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    high_importance_features = ['Liquidity_Z', 'LGC_distance_z', 'SMA_ratio_52w', 'Realised_to_Spot']
    all_predictors = sorted(list(set(features_df.columns) - {'Target', 'Target_12w', 'close_usd'}))

    if forecast_date:
        ref_date = pd.to_datetime(forecast_date, utc=True)
        try:
            idx = features_df.index.get_indexer([ref_date], method='nearest')[0]
            ref_date = features_df.index[idx]
            latest_for_lstm = features_df.iloc[max(0, idx - 3):idx + 1]
            latest_flat = features_df.iloc[[idx]]
        except Exception:
            print(f"❌ Could not find data for forecast date {forecast_date}.")
            return
    else:
        latest_for_lstm = features_df.tail(4)
        latest_flat = features_df.tail(1)
        ref_date = latest_flat.index[0]
        
    last_close_price = latest_flat['close_usd'].iloc[0]

    # --- 3. Prepare Data for Prediction ---
    X_cls_4w_scaled = scaler_cls_4w.transform(latest_flat[high_importance_features])
    X_cls_12w_scaled = scaler_cls_12w.transform(latest_flat[high_importance_features])
    X_reg_4w_scaled = scaler_reg_4w.transform(latest_for_lstm[all_predictors])
    X_latest_seq = np.array([X_reg_4w_scaled])
    X_reg_12w_scaled = scaler_reg_12w.transform(latest_flat[all_predictors])
    
    # --- 4. Generate Forecasts ---
    pred_4w = classifier_4w.predict(X_cls_4w_scaled)[0]
    prob_4w = classifier_4w.predict_proba(X_cls_4w_scaled)[0]
    outlook_4w = "UP" if pred_4w == 1 else "NOT UP"
    confidence_4w = prob_4w[1] if pred_4w == 1 else prob_4w[0]
    
    pred_12w = classifier_12w.predict(X_cls_12w_scaled)[0]
    prob_12w = classifier_12w.predict_proba(X_cls_12w_scaled)[0]
    outlook_12w = "UP" if pred_12w == 1 else "NOT UP"
    confidence_12w = prob_12w[1] if pred_12w == 1 else prob_12w[0]

    return_4w = price_reg_4w.predict(X_latest_seq, verbose=0)[0][0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    return_12w = price_reg_12w.predict(X_reg_12w_scaled)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Calculate Forecast Ranges ---
    y_4w_full = features_df['Target'].dropna()
    X_4w_full_scaled = scaler_reg_4w.transform(features_df.loc[y_4w_full.index][all_predictors])
    X_seq_full, y_seq_full = create_sequences(pd.DataFrame(X_4w_full_scaled, index=y_4w_full.index), y_4w_full)
    preds_4w = price_reg_4w.predict(X_seq_full, verbose=0).flatten()
    errors_4w = y_seq_full - preds_4w
    range_mod_4w = np.percentile(np.abs(errors_4w), 80)
    
    y_12w_full = features_df['Target_12w'].dropna()
    X_12w_full_scaled = scaler_reg_12w.transform(features_df.loc[y_12w_full.index][all_predictors])
    preds_12w = price_reg_12w.predict(X_12w_full_scaled)
    errors_12w = y_12w_full - preds_12w
    range_mod_12w = np.percentile(np.abs(errors_12w), 80)

    # --- 6. Assemble and Print Final Forecast ---
    regime_status = "Risk-Off"
    if 'Liquidity_Z' in latest_flat.columns and 'DXY_26w_trend' in latest_flat.columns:
        if not latest_flat.empty and latest_flat['Liquidity_Z'].iloc[0] > 0 and latest_flat['DXY_26w_trend'].iloc[0] < 1:
            regime_status = "Risk-On"

    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {ref_date.strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    
    date_4w = ref_date + timedelta(weeks=4)
    print(f"\n--- 4-Week Outlook (for {date_4w.strftime('%Y-%m-%d')}) ---")
    print(f"Directional Signal: {outlook_4w} (Confidence: {confidence_4w:.2%})")
    print(f"Price Target (LSTM): ${price_target_4w:,.2f}")
    print(f"Likely Range (80% confidence): ${price_target_4w * (1 - range_mod_4w):,.2f} - ${price_target_4w * (1 + range_mod_4w):,.2f}")
    
    date_12w = ref_date + timedelta(weeks=12)
    print(f"\n--- 12-Week Outlook (for {date_12w.strftime('%Y-%m-%d')}) ---")
    print(f"Directional Signal: {outlook_12w} (Confidence: {confidence_12w:.2%})")
    print(f"Price Target (Lasso): ${price_target_12w:,.2f}")
    print(f"Likely Range (80% confidence): ${price_target_12w * (1 - range_mod_12w):,.2f} - ${price_target_12w * (1 + range_mod_12w):,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
