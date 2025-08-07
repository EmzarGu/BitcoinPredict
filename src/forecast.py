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
    Generates a full, unified forecast using the LSTM and Bayesian models.
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
        # Load the new LSTM, scaler, and the 12-week Bayesian model
        price_target_4w_model = load_model(models_dir / "price_target_4w_lstm.h5")
        scaler_X = joblib.load(models_dir / "scaler_X_4w.joblib")
        price_target_12w_model = joblib.load(models_dir / "price_target_12w_bayes.joblib")
    except Exception as e:
        print(f"❌ A model or scaler could not be loaded: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    if forecast_date:
        target_date = pd.to_datetime(forecast_date, utc=True)
        # Ensure we have enough historical data for the LSTM sequence
        end_loc = features_df.index.get_loc(target_date)
        latest_features = features_df.iloc[end_loc-4:end_loc+1]
    else:
        latest_features = features_df.tail(5) # Need last 4 rows for sequence + current
        
    last_known_features = latest_features.tail(1)
    last_close_price = last_known_features['close_usd'].iloc[0]

    # --- 3. Prepare Data for LSTM ---
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in latest_features.columns]
    
    # Scale the features using the loaded scaler
    scaled_features = scaler_X.transform(latest_features[predictor_cols_exist])
    
    # Create the sequence for prediction
    X_latest_seq = np.array([scaled_features])

    # --- 4. Generate the Ensemble Forecast ---
    X_latest_flat = last_known_features[predictor_cols_exist]
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

    # 4-Week Forecast using LSTM
    return_4w = price_target_4w_model.predict(X_latest_seq)[0][0]
    price_target_4w = last_close_price * (1 + return_4w)
    
    # 12-Week Forecast using BayesianRidge
    return_12w = price_target_12w_model.predict(X_latest_flat)[0]
    price_target_12w = last_close_price * (1 + return_12w)
    
    # --- 5. Assemble and Print Final Forecast ---
    # Note: Calculating a reliable range for the LSTM is more complex.
    # For now, we'll present the point forecast.
    print("\n--- Final, Unified Forecast ---")
    print(f"Reference Week: {last_known_features.index[0].strftime('%Y-%m-%d')}")
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
