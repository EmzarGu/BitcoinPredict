import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def generate_forecast():
    """
    Loads the latest features and trained models to generate the weekly forecast.
    """
    print("--- Generating Weekly Bitcoin Forecast ---")

    # --- 1. Load Models and Latest Features ---
    models_dir = Path("artifacts/models")
    features_path = Path("artifacts/features_latest.parquet")

    if not all([
        (models_dir / "direction_classifier.joblib").exists(),
        (models_dir / "level_forecaster_bayesian.joblib").exists(),
        (models_dir / "level_forecaster_xgboost.joblib").exists(),
        features_path.exists()
    ]):
        print("❌ Error: Models or latest features file not found.")
        print("Please run `src/features.py` and `src/train.py` first.")
        return

    classifier = joblib.load(models_dir / "direction_classifier.joblib")
    bayesian_model = joblib.load(models_dir / "level_forecaster_bayesian.joblib")
    xgboost_regressor = joblib.load(models_dir / "level_forecaster_xgboost.joblib")
    
    latest_features = pd.read_parquet(features_path)
    print("✅ Models and latest features loaded successfully.")

    # Prepare features for prediction (drop the target column if it's present)
    if 'Target' in latest_features.columns:
        X_latest = latest_features.drop(columns=['Target'])
    else:
        X_latest = latest_features

    # --- 2. Generate Forecast Components ---
    
    # Directional Outlook
    direction_probabilities = classifier.predict_proba(X_latest)[0]
    direction_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    predicted_class_index = np.argmax(direction_probabilities)
    direction = direction_map[predicted_class_index]

    # Price Target Forecasts
    bayesian_pred = bayesian_model.predict(X_latest)[0]
    xgboost_pred = xgboost_regressor.predict(X_latest)[0]
    
    # Blended Target (as per tech design)
    blended_return = (bayesian_pred + xgboost_pred) / 2
    last_close_price = X_latest['close_usd'].iloc[0]
    price_target = last_close_price * (1 + blended_return)

    # Price Range Forecast (using Bayesian model's uncertainty)
    _, std_dev = bayesian_model.predict(X_latest, return_std=True)
    std_dev = std_dev[0]
    # Calculate 20th and 80th percentiles (approx. +/- 0.84 std deviations for a normal distribution)
    lower_bound_return = blended_return - (0.84 * std_dev)
    upper_bound_return = blended_return + (0.84 * std_dev)
    price_range = [
        last_close_price * (1 + lower_bound_return),
        last_close_price * (1 + upper_bound_return)
    ]

    # --- 3. Assemble and Print Final Forecast ---
    
    forecast = {
        "reference_week": X_latest.index[0].strftime('%Y-%m-%d'),
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
