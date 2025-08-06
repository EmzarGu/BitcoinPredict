import pandas as pd
import xgboost as xgb
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
    Trains all predictive models, including the new Quantile models for ranges.
    """
    print("--- Starting model training ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    y_4w_df = features_df[['Target']].dropna()
    y_12w_df = features_df[['Target_12w']].dropna()
    
    X_full = features_df.drop(columns=['Target', 'Target_12w'])

    # --- Train 4-Week Models ---
    print("\n--- Training 4-Week Models ---")
    X_4w = X_full.loc[y_4w_df.index]
    y_4w = y_4w_df['Target']

    # Classifier
    def classify_return(ret):
        if ret > 0.05: return 2
        elif ret < -0.05: return 0
        else: return 1
    y_4w_class = y_4w.apply(classify_return)
    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    classifier.fit(X_4w, y_4w_class)
    print("✅ 4-Week Direction Classifier training complete.")

    # Price Target Model
    xgboost_regressor_4w = xgb.XGBRegressor(objective='reg:squarederror')
    xgboost_regressor_4w.fit(X_4w, y_4w)
    print("✅ 4-Week Price Target (XGBoost) model training complete.")

    # Quantile Models for Range
    lower_bound_4w = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.2)
    lower_bound_4w.fit(X_4w, y_4w)
    print("✅ 4-Week Lower Bound (Quantile) model training complete.")
    
    upper_bound_4w = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.8)
    upper_bound_4w.fit(X_4w, y_4w)
    print("✅ 4-Week Upper Bound (Quantile) model training complete.")

    # --- Train 12-Week Models ---
    print("\n--- Training 12-Week Models ---")
    X_12w = X_full.loc[y_12w_df.index]
    y_12w = y_12w_df['Target_12w']

    # Price Target Model
    xgboost_regressor_12w = xgb.XGBRegressor(objective='reg:squarederror')
    xgboost_regressor_12w.fit(X_12w, y_12w)
    print("✅ 12-Week Price Target (XGBoost) model training complete.")

    # Quantile Models for Range
    lower_bound_12w = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.2)
    lower_bound_12w.fit(X_12w, y_12w)
    print("✅ 12-Week Lower Bound (Quantile) model training complete.")
    
    upper_bound_12w = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.8)
    upper_bound_12w.fit(X_12w, y_12w)
    print("✅ 12-Week Upper Bound (Quantile) model training complete.")

    # --- Save All Models ---
    print("\n--- Saving all models to artifacts/models/ ---")
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, models_dir / "direction_classifier.joblib")
    joblib.dump(xgboost_regressor_4w, models_dir / "price_target_4w.joblib")
    joblib.dump(lower_bound_4w, models_dir / "lower_bound_4w.joblib")
    joblib.dump(upper_bound_4w, models_dir / "upper_bound_4w.joblib")
    joblib.dump(xgboost_regressor_12w, models_dir / "price_target_12w.joblib")
    joblib.dump(lower_bound_12w, models_dir / "lower_bound_12w.joblib")
    joblib.dump(upper_bound_12w, models_dir / "upper_bound_12w.joblib")
    
    print("✅ All models saved successfully.")

if __name__ == "__main__":
    train_models()
