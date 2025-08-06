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

def generate_forecast(forecast_date: str = None):
    """
    Loads features and models to generate a full 4-week and 12-week forecast.
    """
    print("--- Generating Full Bitcoin Forecast (4-week & 12-week) ---")

    # --- 1. Load All Models and Features ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Error: Could not build features.")
        return

    if forecast_date:
        try:
            target_date = pd.to_datetime(forecast_date, utc=True)
            time_diffs = np.abs(features_df.index - target_date)
            closest_date_index = time_diffs.argmin()
            latest_features = features_df.iloc[[closest_date_index]]
            print(f"✅ Found closest available data for forecast: {latest_features.index[0].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"❌ Error processing date: {e}")
            return
    else:
        training_features = build_features(for_training=True)
        latest_features = training_features.tail(1)
        print(f"✅ Generating forecast for the latest available week: {latest_features.index[0].strftime('%Y-%m-%d')}")

    classifier = joblib.load(models_dir / "direction_classifier.joblib")
    bayesian_4w = joblib.load(models_dir / "level_forecaster_bayesian_4w.joblib")
    xgboost_4w = joblib.load(models_dir / "level_forecaster_xgboost_4w.joblib")
    bayesian_12w = joblib.load(models_dir / "level_forecaster_bayesian_12w.joblib")
    xgboost_12w = joblib.load(models_dir / "level_forecaster_xgboost_12w.joblib")
    
    training_cols = classifier.get_booster().feature_names
    X_latest = latest_features[training_cols]

    # --- 2. Implement Liquidity Regime Filter ---
    features_df['Liquidity_Z'] = -features_df['dxy_z']
    features_df['DXY_26w_MA'] = features_df['dxy'].rolling(window=26).mean()
    features_df['DXY_26w_trend'] = features_df['dxy'] / features_df['DXY_26w_MA']
    features_df['Risk-On'] = np.where((features_df['Liquidity_Z'] > 0) & (features_df['DXY_26w_trend'] < 1), True, False)
    
    if pd.isna(features_df.loc[latest_features.index, 'Risk-On'].iloc[0]):
        regime_status = "Unknown (Insufficient Data)"
        current_regime_is_risk_on = True
    else:
        current_regime_is_risk_on = features_df.loc[latest_features.index, 'Risk-On'].iloc[0]
        regime_status = "Risk-On" if current_regime_is_risk_on else "Risk-Off"
    
    print(f"✅ Macro Regime detected: {regime_status}")

    # --- 3. Generate 4-Week Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest)[0]
    
    if not current_regime_is_risk_on:
        print("... Adjusting forecast for Risk-Off conditions ...")
        bullish_prob = direction_probabilities[2]
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction
    
    direction_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    direction = direction_map[np.argmax(direction_probabilities)]

    bayesian_pred_4w, std_dev_4w = bayesian_4w.predict(X_latest, return_std=True)
    xgboost_pred_4w = xgboost_4w.predict(X_latest)[0]
    blended_return_4w = (bayesian_pred_4w[0] + xgboost_pred_4w) / 2
    last_close_price = X_latest['close_usd'].iloc[0]
    price_target_4w = last_close_price * (1 + blended_return_4w)

    lower_bound_4w = last_close_price * (1 + (blended_return_4w - (0.84 * std_dev_4w[0])))
    upper_bound_4w = last_close_price * (1 + (blended_return_4w + (0.84 * std_dev_4w[0])))
    
    # --- 4. Generate 12-Week Forecast ---
    bayesian_pred_12w, std_dev_12w = bayesian_12w.predict(X_latest, return_std=True)
    xgboost_pred_12w = xgboost_12w.predict(X_latest)[0]
    blended_return_12w = (bayesian_pred_12w[0] + xgboost_pred_12w) / 2
    price_target_12w = last_close_price * (1 + blended_return_12w)

    lower_bound_12w = last_close_price * (1 + (blended_return_12w - (0.84 * std_dev_12w[0])))
    upper_bound_12w = last_close_price * (1 + (blended_return_12w + (0.84 * std_dev_12w[0])))

    # --- THIS IS THE FIX: Ensure the price floor is $0 ---
    lower_bound_4w = max(0, lower_bound_4w)
    lower_bound_12w = max(0, lower_bound_12w)
    # --------------------------------------------------

    # --- 5. Assemble and Print Final Forecast ---
    print("\n--- Final Forecast ---")
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
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest available data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
