import pandas as pd
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
import joblib
from pathlib import Path
import sys

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features

def train_models():
    """
    Trains all predictive models for both 4-week and 12-week targets
    and saves them to the artifacts directory.
    """
    print("--- Starting model training ---")

    # 1. Load the feature-engineered data
    features_df = build_features()
    if features_df.empty:
        print("❌ Could not build features. Aborting training.")
        return

    print(f"✅ Features loaded successfully with {len(features_df)} records.")

    # --- Prepare Predictor Features (X) ---
    # Predictors are all columns except the two target variables
    predictor_cols = [col for col in features_df.columns if "Target" not in col]
    X_full = features_df[predictor_cols]

    # --- Train 4-Week Models ---
    print("\n--- Training 4-Week Models ---")
    
    # Prepare 4-week target data, dropping rows where it's NaN
    y4w_df = features_df[['Target']].dropna()
    X_4w = X_full.loc[y4w_df.index]
    y_4w = y4w_df['Target']
    
    # Direction Classifier
    def classify_return(ret):
        if ret > 0.05: return 2
        elif ret < -0.05: return 0
        else: return 1
    y_4w_class = y_4w.apply(classify_return)
    
    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    classifier.fit(X_4w, y_4w_class)
    print("✅ 4-Week Direction Classifier training complete.")

    # Bayesian Ridge Regressor
    bayesian_model_4w = BayesianRidge()
    bayesian_model_4w.fit(X_4w, y_4w)
    print("✅ 4-Week Bayesian Ridge model training complete.")

    # XGBoost Regressor
    xgboost_regressor_4w = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mape')
    xgboost_regressor_4w.fit(X_4w, y_4w)
    print("✅ 4-Week XGBoost Regressor model training complete.")

    # --- Train 12-Week Models ---
    print("\n--- Training 12-Week Models ---")

    # Prepare 12-week target data, dropping rows where it's NaN
    y12w_df = features_df[['Target_12w']].dropna()
    X_12w = X_full.loc[y12w_df.index]
    y_12w = y12w_df['Target_12w']

    # Bayesian Ridge Regressor
    bayesian_model_12w = BayesianRidge()
    bayesian_model_12w.fit(X_12w, y_12w)
    print("✅ 12-Week Bayesian Ridge model training complete.")

    # XGBoost Regressor
    xgboost_regressor_12w = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mape')
    xgboost_regressor_12w.fit(X_12w, y_12w)
    print("✅ 12-Week XGBoost Regressor model training complete.")

    # --- Save All Models ---
    print("\n--- Saving all models to artifacts/models/ ---")
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 4-Week Models
    joblib.dump(classifier, models_dir / "direction_classifier.joblib")
    joblib.dump(bayesian_model_4w, models_dir / "level_forecaster_bayesian_4w.joblib")
    joblib.dump(xgboost_regressor_4w, models_dir / "level_forecaster_xgboost_4w.joblib")
    
    # 12-Week Models
    joblib.dump(bayesian_model_12w, models_dir / "level_forecaster_bayesian_12w.joblib")
    joblib.dump(xgboost_regressor_12w, models_dir / "level_forecaster_xgboost_12w.joblib")
    
    print("✅ All models saved successfully.")

if __name__ == "__main__":
    train_models()
