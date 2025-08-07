import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
import argparse
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features, PREDICTOR_COLS
from src.train import create_sequences # Import the helper function

def generate_forecast(forecast_date: str = None):
    """
    Generates a full, unified forecast using the LSTM and Bayesian models,
    including price target ranges.
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
        price_target_4w_model = load_model(models_dir / "price_target_4w_lstm.h5")
        scaler_X = joblib.load(models_dir / "scaler_X_4w.joblib")
        price_target_12w_model = joblib.load(models_dir / "price_target_12w_bayes.joblib")
    except Exception as e:
        print(f"❌ A model or scaler could not be loaded: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in features_df.columns]
    
    if forecast_date:
        target_date = pd.to_datetime(forecast_date, utc=True)
        end_loc = features_df.index.get_loc(target_date)
        # Ensure we have enough historical data for the LSTM sequence
        latest_features = features_df.iloc[max(0, end_loc - 4):end_loc + 1]
    else:
        latest_features = features_df.tail(5)
        
    last_known_features = latest_features.tail(1)
    last_close_price = last_known_features['close_usd'].iloc[0]

    # --- 3. Prepare Data for Prediction ---
    # For LSTM (4-week)
    scaled_features = scaler_X.transform(latest_features[predictor_cols_exist])
    X_latest_seq = np.array([scaled_features])
    
    # For Bayesian (12-week) and Classifier
    X_latest_flat = last_known_features[predictor_cols_exist]

    # --- 4. Generate the Ensemble Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest_flat)[0]

    # Implement Liquidity Regime Filter
    regime_status = "Risk-Off"
    if 'Liquidity_Z' in X_latest_flat.columns and 'DXY_26w_trend' in X_latest_flat.columns:
        if not X_latest_flat.empty and X_latest_flat['Liquidity_Z'].iloc[0] > 0 and X_latest_flat['DXY_26w_trend'].iloc[0] < 1:
            regime_status = "Risk-On"

    if regime_status == "Risk-Off":
        bullish_prob = direction_probabilities[2]
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction * 0.5
        direction_probabilities[0] += reduction * 0.5

    direction = ["Bearish", "Neutral", "Bullish"][np.argmax(direction_probabilities)]

    # 4-Week Forecast (LSTM)
    return_4w = price_target_4w_model.predict(X_latest_seq, verbose=0)[0][0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    # 12-Week Forecast (Bayesian)
    return_12w = price_target_12w_model.predict(X_latest_flat)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Calculate Forecast Ranges using Historical Errors ---
    # 4-Week Range (for LSTM)
    y_4w_full = features_df['Target'].dropna()
    X_4w_full_scaled = scaler_X.transform(features_df.loc[y_4w_full.index][predictor_cols_exist])
    X_seq_full, y_seq_full = create_sequences(pd.DataFrame(X_4w_full_scaled), y_4w_full)
    
    predictions_4w = price_target_4w_model.predict(X_seq_full, verbose=0).flatten()
    errors_4w = y_seq_full - predictions_4w
    range_modifier_4w = np.percentile(np.abs(errors_4w), 80)
    lower_bound_4w = price_target_4w * (1 - range_modifier_4w)
    upper_bound_4w = price_target_4w * (1 + range_modifier_4w)
    
    # 12-Week Range (for Bayesian)
    y_12w_full = features_df['Target_12w'].dropna()
    X_12w_full = features_df.loc[y_12w_full.index][predictor_cols_exist]
    predictions_12w = price_target_12w_model.predict(X_12w_full)
    errors_12w = y_12w_full - predictions_12w
    range_modifier_12w = np.percentile(np.abs(errors_12w), 80)
    lower_bound_12w = price_target_12w * (1 - range_modifier_12w)
    upper_bound_12w = price_target_12w * (1 + range_modifier_12w)

    # --- 6. Assemble and Print Final Forecast ---
    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {last_known_features.index[0].strftime('%Y-%m-%d')}")
    print(f"Last Known Price: ${last_close_price:,.2f}")
    print(f"Macro Regime: {regime_status}")
    print(f"Directional Outlook: {direction} (Probabilities: Bearish {direction_probabilities[0]:.2%}, Neutral {direction_probabilities[1]:.2%}, Bullish {direction_probabilities[2]:.2%})")
    print("\n--- 4-Week Outlook (LSTM) ---")
    print(f"Price Target: ${price_target_4w:,.2f}")
    print(f"Likely Range (80% confidence): ${lower_bound_4w:,.2f} - ${upper_bound_4w:,.2f}")
    print("\n--- 12-Week Outlook (Bayesian) ---")
    print(f"Price Target: ${price_target_12w:,.2f}")
    print(f"Likely Range (80% confidence): ${lower_bound_12w:,.2f} - ${upper_bound_12w:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
