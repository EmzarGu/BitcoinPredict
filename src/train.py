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

from src.features import build_features, FEATURE_COLS

def train_regime_models(features_df: pd.DataFrame, regime_name: str):
    """
    Trains a full set of specialist models for a specific market regime.
    """
    print(f"\n--- Training Specialist Models for: {regime_name} Regime ---")
    
    # --- THIS IS THE FIX ---
    # We select only the official predictor columns from the original features.py
    predictor_cols = [col for col in FEATURE_COLS if "Target" not in col]
    X_full = features_df[predictor_cols]
    
    y_4w_df = features_df[['Target']].dropna()
    X_4w = X_full.loc[y_4w_df.index]
    y_4w = y_4w_df['Target']
    def classify(ret): return 2 if ret > 0.05 else (0 if ret < -0.05 else 1)
    y_4w_class = y_4w.apply(classify)

    y_12w_df = features_df[['Target_12w']].dropna()
    X_12w = X_full.loc[y_12w_df.index]
    y_12w = y_12w_df['Target_12w']

    # --- Train and Evaluate Models ---
    print(f"--- Performance Report for {regime_name} Regime ---")
    
    # Split data for evaluation
    if len(X_4w) < 10 or len(X_12w) < 10: # Check if there's enough data
        print("Insufficient data to create a test split. Skipping evaluation.")
        # If no test split, train on all data for the regime
        X_train_4w, y_train_4w_class, y_train_4w = X_4w, y_4w_class, y_4w
        X_train_12w, y_train_12w = X_12w, y_12w
    else:
        X_train_4w, X_test_4w, y_train_4w_class, y_test_4w_class = train_test_split(X_4w, y_4w_class, test_size=0.2, shuffle=False)
        _, _, y_train_4w, y_test_4w = train_test_split(X_4w, y_4w, test_size=0.2, shuffle=False)
        X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_12w, y_12w, test_size=0.2, shuffle=False)

        # Direction Classifier
        classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
        classifier.fit(X_train_4w, y_train_4w_class)
        accuracy = accuracy_score(y_test_4w_class, classifier.predict(X_test_4w))
        print(f"✅ Directional Classifier Accuracy: {accuracy:.2%}")

        # 4-Week Regressor
        xgboost_4w = xgb.XGBRegressor(objective='reg:squarederror')
        xgboost_4w.fit(X_train_4w, y_train_4w)
        mape_4w = mean_absolute_percentage_error(y_test_4w, xgboost_4w.predict(X_test_4w))
        print(f"✅ 4-Week Price Target MAPE: {mape_4w:.2%}")
        
        # 12-Week Regressor
        xgboost_12w = xgb.XGBRegressor(objective='reg:squarederror')
        xgboost_12w.fit(X_train_12w, y_train_12w)
        mape_12w = mean_absolute_percentage_error(y_test_12w, xgboost_12w.predict(X_test_12w))
        print(f"✅ 12-Week Price Target MAPE: {mape_12w:.2%}")

    # --- Re-train on all data for this regime and save ---
    final_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3).fit(X_4w, y_4w_class)
    final_xgboost_4w = xgb.XGBRegressor(objective='reg:squarederror').fit(X_4w, y_4w)
    final_xgboost_12w = xgb.XGBRegressor(objective='reg:squarederror').fit(X_12w, y_12w)
    
    models_dir = Path(f"artifacts/models/{regime_name.lower()}")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_classifier, models_dir / "direction_classifier.joblib")
    joblib.dump(final_xgboost_4w, models_dir / "price_target_4w.joblib")
    joblib.dump(final_xgboost_12w, models_dir / "price_target_12w.joblib")
    print(f"✅ Specialist models for {regime_name} Regime saved successfully.")

def train_all_models():
    """
    Trains specialist models for different market regimes.
    """
    print("--- Starting Regime-Specific Model Training ---")

    # 1. Load Data
    features_df = build_features(for_training=False)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Define Market Regimes ---
    features_df['52w_MA'] = features_df['close_usd'].rolling(window=52).mean()
    features_df['Regime'] = np.where(features_df['close_usd'] > features_df['52w_MA'], 'Bull', 'Bear')
    
    print("✅ Market regimes defined successfully.")
    print(features_df['Regime'].value_counts())

    # --- 3. Train Models for Each Regime ---
    for regime in ['Bull', 'Bear']:
        regime_df = features_df[features_df['Regime'] == regime].copy()
        if len(regime_df) > 50:
             train_regime_models(regime_df, regime)
        else:
             print(f"\n--- Insufficient data to train specialist models for: {regime} Regime ---")

if __name__ == "__main__":
    train_all_models()
