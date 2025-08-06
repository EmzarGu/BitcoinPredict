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
    Loads features and models to generate a forecast for a specific date.
    If no date is provided, it forecasts for the latest available date.
    """
    print("--- Generating Bitcoin Forecast ---")

    # --- 1. Load Models and Features ---
    models_dir = Path("artifacts/models")
    features_df = build_features()

    if features_df.empty:
        print("❌ Error: Could not build features.")
        return

    # --- THIS IS THE CORRECTED DATE LOOKUP LOGIC ---
    if forecast_date:
        try:
            target_date = pd.to_datetime(forecast_date, utc=True)
            # Calculate the absolute difference between the target date and all index dates
            time_diff = (features_df.index - target_date).to_series().abs()
            # Find the index of the row with the minimum time difference
            closest_date_index = time_diff.idxmin()
            latest_features = features_df.loc[[closest_date_index]]
            print(f"✅ Found closest available data for forecast: {latest_features.index[0].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"❌ Error processing date: {e}")
            return
    else:
        latest_features = features_df.tail(1)
        print(f"✅ Generating forecast for the latest available week: {latest_features.index[0].strftime('%Y-%m-%d')}")

    classifier = joblib.load(models_dir / "direction_classifier.joblib")
    bayesian_model = joblib.load(models_dir / "level_forecaster_bayesian.joblib")
    xgboost_regressor = joblib.load(models_dir / "level_forecaster_xgboost.joblib")
    
    X_latest = latest_features.drop(columns=['Target'])

    # --- 2. Implement Liquidity Regime Filter ---
    features_df['Liquidity_Z'] = -features_df['dxy_z']
    features_df['DXY_26w_MA'] = features_df['dxy'].rolling(window=26).mean()
    features_df['DXY_26w_trend'] = features_df['dxy'] / features_df['DXY_26w_MA']
    
    features_df['Risk-On'] = np.where(
        (features_df['Liquidity_Z'] > 0) & (features_df['DXY_26w_trend'] < 1),
        True, False
    )
    
    current_regime_is_risk_on = features_df.loc[latest_features.index, 'Risk-On'].iloc[0]
    regime_status = "Risk-On" if current_regime_is_risk_on else "Risk-Off"
    print(f"✅ Macro Regime detected: {regime_status}")

    # --- 3. Generate and Adjust Forecast ---
    direction_probabilities = classifier.predict_proba(X_latest)[0]
    
    if not current_regime_is_risk_on:
        print("... Adjusting forecast for Risk-Off conditions ...")
        bullish_prob = direction_probabilities[2]
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction
    
    direction_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    predicted_class_index = np.argmax(direction_probabilities)
    direction = direction_map[predicted_class_index]

    bayesian_pred, std_dev = bayesian_model.predict(X_latest, return_std=True)
    bayesian_pred = bayesian_pred[0]
    std_dev = std_dev[0]
    xgboost_pred = xgboost_regressor.predict(X_latest)[0]
    
    blended_return = (bayesian_pred + xgboost_pred) / 2
    last_close_price = X_latest['close_usd'].iloc[0]
    price_target = last_close_price * (1 + blended_return)

    lower_bound_return = blended_return - (0.84 * std_dev)
    upper_bound_return = blended_return + (0.84 * std_dev)
    price_range = [
        last_close_price * (1 + lower_bound_return),
        last_close_price * (1 + upper_bound_return)
    ]

    # --- 4. Assemble and Print Final Forecast ---
    forecast = {
        "reference_week": latest_features.index[0].strftime('%Y-%m-%d'),
        "last_known_price": f"${last_close_price:,.2f}",
        "macro_regime": regime_status,
        "directional_outlook": direction,
        "probabilities": {
            "Bearish": f"{direction_probabilities[0]:.2%}",
            "Neutral": f"{direction_probabilities[1]:.2%}",
            "Bullish": f"{direction_probabilities[2]:.2%}"
        },
        "4_week_price_target": f"${price_target:,.2f}",
        "4_week_likely_range": f"${price_range[0]:,.2f} - ${price_range[1]:,.2f}"
    }

    print("\n--- Final Forecast ---")
    for key, value in forecast.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Bitcoin forecast for a specific date.')
    parser.add_argument('--date', type=str, help='The date for the forecast in YYYY-MM-DD format. Defaults to the latest available data.')
    args = parser.parse_args()
    
    generate_forecast(args.date)
