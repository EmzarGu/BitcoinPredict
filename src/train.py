import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.linear_model import BayesianRidge
# Note: TensorFlow/Keras for LSTM is a future step

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import from our updated features module
from src.features import build_features, FEATURE_COLS

def train_all_models():
    """
    Trains, evaluates, and saves all predictive models as specified in the blueprint.
    This now includes the Bayesian Ridge Regressor.
    """
    print("--- Starting Model Training & Evaluation ---")

    # 1. Load Data using the updated build_features function
    # Using the FEATURE_COLS list imported from features.py
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Prepare Data ---
    # The predictor columns are now dynamically taken from FEATURE_COLS
    predictor_cols = [col for col in FEATURE_COLS if "Target" not in col]
    X_full = features_df[predictor_cols]

    # 4-Week Data
    y_4w_df = features_df[['Target']].dropna()
    X_4w = X_full.loc[y_4w_df.index]
    y_4w = y_4w_df['Target']
    def classify(ret): return 2 if ret > 0.05 else (0 if ret < -0.05 else 1) # Bullish, Bearish, Neutral
    y_4w_class = y_4w.apply(classify)

    # 12-Week Data
    y_12w_df = features_df[['Target_12w']].dropna()
    X_12w = X_full.loc[y_12w_df.index]
    y_12w = y_12w_df['Target_12w']

    # --- 3. Split Data into Training and Testing Sets ---
    # Using a standard 80/20 time-series split
    X_train_4w, X_test_4w, y_train_4w_class, y_test_4w_class = train_test_split(X_4w, y_4w_class, test_size=0.2, shuffle=False)
    _, _, y_train_4w, y_test_4w = train_test_split(X_4w, y_4w, test_size=0.2, shuffle=False)
    X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_12w, y_12w, test_size=0.2, shuffle=False)

    print(f"✅ Data split into training and testing sets.")

    # --- 4. Train and Evaluate Models on the Split Data ---
    print("\n--- Model Performance Report (on unseen test data) ---")

    # Direction Classifier (XGBoost)
    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    classifier.fit(X_train_4w, y_train_4w_class)
    accuracy = accuracy_score(y_test_4w_class, classifier.predict(X_test_4w))
    print(f"\n✅ Directional Classifier (4-Week):")
    print(f"   - Test Accuracy: {accuracy:.2%}")

    # 4-Week Regressor (Bayesian Ridge) - As per blueprint
    bayesian_4w = BayesianRidge()
    bayesian_4w.fit(X_train_4w, y_train_4w)
    mape_4w_bayes = mean_absolute_percentage_error(y_test_4w, bayesian_4w.predict(X_test_4w))
    print(f"\n✅ Price Target Regressor (4-Week, BayesianRidge):")
    print(f"   - Test MAPE: {mape_4w_bayes:.2%}")

    # 12-Week Regressor (Bayesian Ridge) - As per blueprint
    bayesian_12w = BayesianRidge()
    bayesian_12w.fit(X_train_12w, y_train_12w)
    mape_12w_bayes = mean_absolute_percentage_error(y_test_12w, bayesian_12w.predict(X_test_12w))
    print(f"\n✅ Price Target Regressor (12-Week, BayesianRidge):")
    print(f"   - Test MAPE: {mape_12w_bayes:.2%}")

    # --- 5. Re-train Final Models on All Data and Save ---
    print("\n\n--- Re-training final models on all available data ---")

    # Re-train on the full dataset available for each target
    classifier.fit(X_4w, y_4w_class)
    bayesian_4w.fit(X_4w, y_4w)
    bayesian_12w.fit(X_12w, y_12w)

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the models with new names to reflect the algorithm change
    joblib.dump(classifier, models_dir / "direction_classifier_xgb.joblib")
    joblib.dump(bayesian_4w, models_dir / "price_target_4w_bayes.joblib")
    joblib.dump(bayesian_12w, models_dir / "price_target_12w_bayes.joblib")

    print("✅ All final models trained and saved successfully.")

if __name__ == "__main__":
    train_all_models()
