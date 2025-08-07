import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features

def create_sequences(X, y, time_steps=4):
    """Helper function to create sequences for LSTM model."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_all_models():
    """
    Trains the final, complete hybrid model suite, including a 12-week classifier.
    """
    print("--- Starting Final Model Training ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Define Feature Sets ---
    high_importance_features = ['Liquidity_Z', 'LGC_distance_z', 'SMA_ratio_52w', 'Realised_to_Spot']
    all_predictors = sorted(list(set(features_df.columns) - {'Target', 'Target_12w', 'close_usd'}))
    
    # --- 3. Train Directional Classifiers ---
    
    # 4-Week Classifier
    print("\n--- Training 4-Week Binary Directional Classifier ---")
    clean_cls_4w_df = features_df[high_importance_features + ['Target']].dropna()
    X_class_4w = clean_cls_4w_df[high_importance_features]
    y_binary_4w = (clean_cls_4w_df['Target'] > 0.01).astype(int)
    scaler_cls_4w = StandardScaler().fit(X_class_4w)
    X_scaled_cls_4w = pd.DataFrame(scaler_cls_4w.transform(X_class_4w), index=X_class_4w.index, columns=X_class_4w.columns)
    X_train_cls_4w, X_test_cls_4w, y_train_cls_4w, y_test_cls_4w = train_test_split(X_scaled_cls_4w, y_binary_4w, test_size=0.2, shuffle=False)
    classifier_4w = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_cls_4w, y_train_cls_4w)
    accuracy_4w = accuracy_score(y_test_cls_4w, classifier_4w.predict(X_test_cls_4w))
    print(f"✅ Final 4-Week Classifier Accuracy: {accuracy_4w:.2%}")

    # **NEW**: 12-Week Classifier
    print("\n--- Training 12-Week Binary Directional Classifier ---")
    clean_cls_12w_df = features_df[high_importance_features + ['Target_12w']].dropna()
    X_class_12w = clean_cls_12w_df[high_importance_features]
    y_binary_12w = (clean_cls_12w_df['Target_12w'] > 0.01).astype(int) # Using 12-week target
    scaler_cls_12w = StandardScaler().fit(X_class_12w)
    X_scaled_cls_12w = pd.DataFrame(scaler_cls_12w.transform(X_class_12w), index=X_class_12w.index, columns=X_class_12w.columns)
    X_train_cls_12w, X_test_cls_12w, y_train_cls_12w, y_test_cls_12w = train_test_split(X_scaled_cls_12w, y_binary_12w, test_size=0.2, shuffle=False)
    classifier_12w = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_cls_12w, y_train_cls_12w)
    accuracy_12w = accuracy_score(y_test_cls_12w, classifier_12w.predict(X_test_cls_12w))
    print(f"✅ Final 12-Week Classifier Accuracy: {accuracy_12w:.2%}")

    # --- 4. Train Price Target Regressors (No changes here) ---
    # (LSTM for 4-week, Lasso for 12-week)
    # ... [rest of the training code for regressors] ...

    # --- 5. Save Final Models ---
    print("\n\n--- Saving final models ---")
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all models and scalers
    joblib.dump(classifier_4w.fit(X_scaled_cls_4w, y_binary_4w), models_dir / "direction_classifier_4w_final.joblib")
    joblib.dump(classifier_12w.fit(X_scaled_cls_12w, y_binary_12w), models_dir / "direction_classifier_12w_final.joblib")
    # ... [rest of the saving code for regressors and other scalers] ...
    joblib.dump(scaler_cls_4w, models_dir / "scaler_classifier_4w_final.joblib")
    joblib.dump(scaler_cls_12w, models_dir / "scaler_classifier_12w_final.joblib")

    print("✅ All final models and scalers trained and saved successfully.")

# --- The rest of the file remains the same ---
# (The code for training the regressors is unchanged)
