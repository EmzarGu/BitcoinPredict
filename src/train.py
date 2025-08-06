import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features

def train_models():
    """
    Trains, evaluates, and saves all predictive models using a correct
    time-series split to prevent data leakage.
    """
    print("--- Starting Model Training & Evaluation ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Prepare Data ---
    predictor_cols = [col for col in features_df.columns if "Target" not in col]
    X_full = features_df[predictor_cols]

    # 4-Week Data
    y_4w_df = features_df[['Target']].dropna()
    X_4w = X_full.loc[y_4w_df.index]
    y_4w = y_4w_df['Target']
    def classify(ret): return 2 if ret > 0.05 else (0 if ret < -0.05 else 1)
    y_4w_class = y_4w.apply(classify)

    # 12-Week Data
    y_12w_df = features_df[['Target_12w']].dropna()
    X_12w = X_full.loc[y_12w_df.index]
    y_12w = y_12w_df['Target_12w']

    # --- 3. Split Data into Training and Testing Sets ---
    # We use shuffle=False to respect the time-series nature of the data
    X_train_4w, X_test_4w, y_train_4w, y_test_4w = train_test_split(X_4w, y_4w, test_size=0.2, shuffle=False)
    _, _, y_train_4w_class, y_test_4w_class = train_test_split(X_4w, y_4w_class, test_size=0.2, shuffle=False)
    X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_12w, y_12w, test_size=0.2, shuffle=False)
    
    print(f"✅ Data split into training and testing sets.")

    # --- 4. Train and Evaluate Models on the Split Data ---
    print("\n--- Model Performance Report (on unseen test data) ---")

    # Direction Classifier
    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    classifier.fit(X_train_4w, y_train_4w_class)
    accuracy = accuracy_score(y_test_4w_class, classifier.predict(X_test_4w))
    print(f"\n✅ Directional Classifier (4-Week):")
    print(f"   - True Accuracy: {accuracy:.2%}")

    # 4-Week Regressor
    xgboost_4w = xgb.XGBRegressor(objective='reg:squarederror')
    xgboost_4w.fit(X_train_4w, y_train_4w)
    mape_4w = mean_absolute_percentage_error(y_test_4w, xgboost_4w.predict(X_test_4w))
    print(f"\n✅ Price Target Regressor (4-Week):")
    print(f"   - True MAPE: {mape_4w:.2%}")

    # 12-Week Regressor
    xgboost_12w = xgb.XGBRegressor(objective='reg:squarederror')
    xgboost_12w.fit(X_train_12w, y_train_12w)
    mape_12w = mean_absolute_percentage_error(y_test_12w, xgboost_12w.predict(X_test_12w))
    print(f"\n✅ Price Target Regressor (12-Week):")
    print(f"   - True MAPE: {mape_12w:.2%}")

    # --- 5. Re-train Final Models on All Data and Save ---
    print("\n\n--- Re-training final models on all available data ---")
    
    # Re-train with all available data to make the final models as smart as possible
    classifier.fit(X_4w, y_4w_class)
    xgboost_4w.fit(X_4w, y_4w)
    xgboost_12w.fit(X_12w, y_12w)
    
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, models_dir / "direction_classifier.joblib")
    joblib.dump(xgboost_4w, models_dir / "price_target_4w.joblib")
    joblib.dump(xgboost_12w, models_dir / "price_target_12w.joblib")
    
    print("✅ All final models saved successfully.")

if __name__ == "__main__":
    train_models()
