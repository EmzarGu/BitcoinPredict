import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, BayesianRidge

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features, PREDICTOR_COLS

def train_all_models():
    """
    Trains, evaluates, and saves all predictive models.
    The price target model is now Lasso Regression.
    """
    print("--- Starting Model Training & Evaluation ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Prepare Data ---
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in features_df.columns]
    X_full = features_df[predictor_cols_exist]

    y_4w_df = features_df[['Target']].dropna()
    X_4w_data = X_full.loc[y_4w_df.index]
    y_4w = y_4w_df['Target']

    y_12w_df = features_df[['Target_12w']].dropna()
    X_12w_data = X_full.loc[y_12w_df.index]
    y_12w = y_12w_df['Target_12w']

    # --- 3. Scale Data and Split ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_full.index, columns=X_full.columns)

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled_df.loc[y_4w_df.index], y_4w, test_size=0.2, shuffle=False)
    
    def classify(ret): return 2 if ret > 0.05 else (0 if ret < -0.05 else 1)
    y_train_class = y_train_reg.apply(classify)
    y_test_class = y_test_reg.apply(classify)

    print(f"✅ Data split and scaled.")

    # --- 4. Train and Evaluate Models ---
    print("\n--- Model Performance Report (on unseen test data) ---")
    
    # Directional Classifier (Logistic Regression)
    classifier = LogisticRegression(max_iter=1000, multi_class='multinomial')
    classifier.fit(X_train, y_train_class)
    accuracy = accuracy_score(y_test_class, classifier.predict(X_test))
    print(f"\n✅ Directional Classifier (Logistic Regression):")
    print(f"   - Test Accuracy: {accuracy:.2%}")

    # 4-Week Regressor (Lasso)
    price_reg_4w = Lasso(alpha=0.01) # Using a small alpha to start
    price_reg_4w.fit(X_train, y_train_reg)
    mape_4w = mean_absolute_percentage_error(y_test_reg, price_reg_4w.predict(X_test))
    print(f"\n✅ Price Target Regressor (4-Week, Lasso):")
    print(f"   - Test MAPE: {mape_4w:.2%}")

    # 12-Week Regressor (Lasso)
    X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_scaled_df.loc[y_12w_df.index], y_12w, test_size=0.2, shuffle=False)
    price_reg_12w = Lasso(alpha=0.01)
    price_reg_12w.fit(X_train_12w, y_train_12w)
    mape_12w = mean_absolute_percentage_error(y_test_12w, price_reg_12w.predict(X_test_12w))
    print(f"\n✅ Price Target Regressor (12-Week, Lasso):")
    print(f"   - Test MAPE: {mape_12w:.2%}")

    # --- 5. Re-train Final Models on All Data and Save ---
    print("\n\n--- Re-training final models on all available data ---")
    
    scaler_final = StandardScaler()
    X_scaled_full = scaler_final.fit_transform(X_full)
    X_scaled_full_df = pd.DataFrame(X_scaled_full, index=X_full.index, columns=X_full.columns)
    
    y_full_class = y_4w.apply(classify)
    classifier.fit(X_scaled_full_df.loc[y_4w.index], y_full_class)
    price_reg_4w.fit(X_scaled_full_df.loc[y_4w.index], y_4w)
    price_reg_12w.fit(X_scaled_full_df.loc[y_12w.index], y_12w)

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, models_dir / "direction_classifier_logreg.joblib")
    joblib.dump(price_reg_4w, models_dir / "price_target_4w_lasso.joblib")
    joblib.dump(price_reg_12w, models_dir / "price_target_12w_lasso.joblib")
    joblib.dump(scaler_final, models_dir / "scaler_final.joblib")

    print("✅ All final models trained and saved successfully.")

if __name__ == "__main__":
    train_all_models()
