import pandas as pd
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
import joblib
import os
from pathlib import Path
import sys

# --- THIS IS THE FIX ---
# Add the project's root directory to the Python path
# This allows the script to find the 'src' module
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# ---------------------

# Import the feature building function from your existing script
from src.features import build_features

def train_models():
    """
    Trains all predictive models and saves them to the artifacts directory.
    """
    print("--- Starting model training ---")

    # 1. Load the feature-engineered data
    features_df = build_features()
    if features_df.empty:
        print("❌ Could not build features. Aborting training.")
        return

    print(f"✅ Features loaded successfully with {len(features_df)} records.")

    # --- Prepare Data ---
    y = features_df['Target']
    X = features_df.drop(columns=['Target'])

    # --- Train Direction Classifier (XGBoost) ---
    print("\n--- Training Direction Classifier (XGBoost) ---")
    
    def classify_return(ret):
        if ret > 0.05: return 2  # Bullish
        elif ret < -0.05: return 0  # Bearish
        else: return 1  # Neutral
    
    y_class = y.apply(classify_return)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    classifier.fit(X, y_class)
    print("✅ Classifier training complete.")

    # --- Train Level Forecaster (Bayesian Ridge) ---
    print("\n--- Training Level Forecaster (Bayesian Ridge) ---")
    bayesian_model = BayesianRidge()
    bayesian_model.fit(X, y)
    print("✅ Bayesian Ridge model training complete.")

    # --- Train Level Forecaster (XGBoost Regressor) ---
    print("\n--- Training Level Forecaster (XGBoost Regressor) ---")
    xgboost_regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mape')
    xgboost_regressor.fit(X, y)
    print("✅ XGBoost Regressor model training complete.")

    # --- Save Models ---
    print("\n--- Saving models to artifacts/models/ ---")
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, models_dir / "direction_classifier.joblib")
    joblib.dump(bayesian_model, models_dir / "level_forecaster_bayesian.joblib")
    joblib.dump(xgboost_regressor, models_dir / "level_forecaster_xgboost.joblib")
    
    print("✅ All models saved successfully.")

if __name__ == "__main__":
    train_models()
