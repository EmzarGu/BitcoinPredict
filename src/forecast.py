import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from src.features import build_features

def generate_forecast():
    """
    Loads the latest features and trained models to generate the weekly forecast,
    including the Liquidity Regime Filter adjustment.
    """
    print("--- Generating Weekly Bitcoin Forecast ---")

    # --- 1. Load Models and Features ---
    models_dir = Path("artifacts/models")
    features_df = build_features() # We need the full history to calculate rolling averages

    if features_df.empty:
        print("❌ Error: Could not build features. Aborting forecast.")
        return
        
    latest_features = features_df.tail(1)

    classifier = joblib.load(models_dir / "direction_classifier.joblib")
    bayesian_model = joblib.load(models_dir / "level_forecaster_bayesian.joblib")
    xgboost_regressor = joblib.load(models_dir / "level_forecaster_xgboost.joblib")
    
    print("✅ Models and latest features loaded successfully.")

    X_latest = latest_features.drop(columns=['Target'])

    # --- 2. Implement Liquidity Regime Filter ---
    # We use the full features_df to get the necessary historical data for the rolling average
    features_df['Liquidity_Z'] = -features_df['dxy_z']
    features_df['DXY_26w_MA'] = features_df['dxy'].rolling(window=26).mean()
    features_df['DXY_26w_trend'] = features_df['dxy'] / features_df['DXY_26w_MA']
    
    features_df['Risk-On'] = np.where(
        (features_df['Liquidity_Z'] > 0) & (features_df['DXY_26w_trend'] < 1),
        True, False
    )
    
    # Get the regime for the most recent week
    current_regime_is_risk_on = features_df['Risk-On'].iloc[-1]
    regime_status = "Risk-On" if current_regime_is_risk_on else "Risk-Off"
    print(f"✅ Current Macro Regime detected: {regime_status}")

    # --- 3. Generate Forecast Components ---
    direction_probabilities = classifier.predict_proba(X_latest)[0]
    
    # --- 4. Adjust Forecast Based on Regime ---
    if not current_regime_is_risk_on:
        print("... Adjusting forecast for Risk-Off conditions ...")
        bullish_prob = direction_probabilities[2]
        # Reduce bullish probability by 30% (relative) and redistribute
        reduction = bullish_prob * 0.30
        direction_probabilities[2] -= reduction
        direction_probabilities[1] += reduction # Add the reduction to the neutral probability
    
    direction_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    predicted_class_index = np.argmax(direction_probabilities)
    direction = direction_map[predicted_class_index]

    bayesian_pred = bayesian_model.predict(X_latest)[0]
    xgboost_pred = xgboost_regressor.predict(X_latest)[0]
    
    blended_return = (bayesian_pred + xgboost_pred) / 2
    last_close_price = X_latest['close_usd'].iloc[0]
    price_target = last_close_price * (1 + blended_return)

    _, std_dev = bayesian_model.predict(X_latest, return_std=True)
    std_dev = std_dev[0]
    
    lower_bound_return = blended_return - (0.84 * std_dev)
    upper_bound_return = blended_return + (0.84 * std_dev)
    price_range = [
        last_close_price * (1 + lower_bound_return),
        last_close_price * (1 + upper_bound_return)
    ]

    # --- 5. Assemble and Print Final Forecast ---
    forecast = {
        "reference_week": X_latest.index[0].strftime('%Y-%m-%d'),
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
    generate_forecast()
