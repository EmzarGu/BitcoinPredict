import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
import argparse
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features
from src.train import create_sequences

def generate_forecast(forecast_date: str = None):
    """
    Generates a final, unified forecast including a 12-week directional signal.
    """
    print("--- Generating Unified Bitcoin Forecast ---")

    # --- 1. Load Data and Models ---
    models_dir = Path("artifacts/models")
    features_df = build_features(for_training=False)

    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    try:
        # Load all models, including the new 12-week classifier
        classifier_4w = joblib.load(models_dir / "direction_classifier_4w_final.joblib")
        classifier_12w = joblib.load(models_dir / "direction_classifier_12w_final.joblib")
        price_reg_4w = load_model(models_dir / "price_target_4w_final.h5")
        price_reg_12w = joblib.load(models_dir / "price_target_12w_final.joblib")
        scaler_cls_4w = joblib.load(models_dir / "scaler_classifier_4w_final.joblib")
        scaler_cls_12w = joblib.load(models_dir / "scaler_classifier_12w_final.joblib")
        # ... [rest of the loading code for regressor scalers] ...
    except Exception as e:
        print(f"❌ A model or scaler could not be loaded: {e}. Please run train.py first.")
        return

    # --- 2. Get Latest Features ---
    high_importance_features = ['Liquidity_Z', 'LGC_distance_z', 'SMA_ratio_52w', 'Realised_to_Spot']
    # ... [rest of the feature preparation code] ...

    # --- 3. Prepare Data for Prediction ---
    X_cls_4w_scaled = scaler_cls_4w.transform(latest_flat[high_importance_features])
    X_cls_12w_scaled = scaler_cls_12w.transform(latest_flat[high_importance_features])
    # ... [rest of the scaling code for regressors] ...
    
    # --- 4. Generate Forecasts ---
    # 4-Week Direction
    pred_4w = classifier_4w.predict(X_cls_4w_scaled)[0]
    prob_4w = classifier_4w.predict_proba(X_cls_4w_scaled)[0]
    outlook_4w = "UP" if pred_4w == 1 else "NOT UP"
    confidence_4w = prob_4w[1] if pred_4w == 1 else prob_4w[0]

    # **NEW**: 12-Week Direction
    pred_12w = classifier_12w.predict(X_cls_12w_scaled)[0]
    prob_12w = classifier_12w.predict_proba(X_cls_12w_scaled)[0]
    outlook_12w = "UP" if pred_12w == 1 else "NOT UP"
    confidence_12w = prob_12w[1] if pred_12w == 1 else prob_12w[0]

    # ... [rest of the price target forecast code] ...
    
    # --- 5. Assemble and Print Final Forecast ---
    # ... [Header of the forecast printout] ...
    
    date_4w = ref_date + timedelta(weeks=4)
    print(f"\n--- 4-Week Outlook (for {date_4w.strftime('%Y-%m-%d')}) ---")
    print(f"Directional Signal: {outlook_4w} (Confidence: {confidence_4w:.2%})")
    # ... [rest of the 4-week printout] ...
    
    date_12w = ref_date + timedelta(weeks=12)
    print(f"\n--- 12-Week Outlook (for {date_12w.strftime('%Y-%m-%d')}) ---")
    print(f"Directional Signal: {outlook_12w} (Confidence: {confidence_12w:.2%})") # Added
    # ... [rest of the 12-week printout] ...

# --- The rest of the file remains the same ---
