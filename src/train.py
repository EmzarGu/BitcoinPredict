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
    Trains the final, complete hybrid model suite.
    """
    print("--- Starting Final Model Training ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Define Feature Sets and Prepare Data ---
    high_importance_features = ['Liquidity_Z', 'LGC_distance_z', 'SMA_ratio_52w', 'Realised_to_Spot']
    # **THIS IS THE FIX**: Sort the column names to ensure consistent order
    all_predictors = sorted(list(set(features_df.columns) - {'Target', 'Target_12w', 'close_usd'}))
    
    # --- 3. Train Binary Directional Classifier ---
    print("\n--- Training Binary Directional Classifier ---")
    clean_cls_df = features_df[high_importance_features + ['Target']].dropna()
    X_class = clean_cls_df[high_importance_features]
    y_binary = (clean_cls_df['Target'] > 0.01).astype(int)
    
    scaler_cls = StandardScaler().fit(X_class)
    X_scaled_cls = pd.DataFrame(scaler_cls.transform(X_class), index=X_class.index, columns=X_class.columns)

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled_cls, y_binary, test_size=0.2, shuffle=False)
    
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_cls, y_train_cls)
    accuracy = accuracy_score(y_test_cls, classifier.predict(X_test_cls))
    print(f"✅ Final Binary Classifier Accuracy: {accuracy:.2%}")

    # --- 4. Train Price Target Regressors ---
    
    # 4-Week Regressor (LSTM)
    print("\n--- Training 4-Week Price Target Regressor (LSTM) ---")
    clean_reg_4w_df = features_df[all_predictors + ['Target']].dropna()
    X_reg_4w = clean_reg_4w_df[all_predictors]
    y_reg_4w = clean_reg_4w_df['Target']
    
    scaler_reg_4w = StandardScaler().fit(X_reg_4w)
    X_scaled_reg_4w = pd.DataFrame(scaler_reg_4w.transform(X_reg_4w), index=X_reg_4w.index, columns=X_reg_4w.columns)
    
    X_train_4w, X_test_4w, y_train_4w, y_test_4w = train_test_split(X_scaled_reg_4w, y_reg_4w, test_size=0.2, shuffle=False)
    time_steps = 4
    X_train_seq, y_train_seq = create_sequences(X_train_4w, y_train_4w, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_4w, y_test_4w, time_steps)

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm_model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32,
                   validation_data=(X_test_seq, y_test_seq), callbacks=[early_stopping], verbose=0)
    predictions_4w = lstm_model.predict(X_test_seq)
    mape_4w = mean_absolute_percentage_error(y_test_seq, predictions_4w)
    print(f"✅ Final 4-Week LSTM MAPE: {mape_4w:.2%}")

    # 12-Week Regressor (Lasso)
    print("\n--- Training 12-Week Price Target Regressor (Lasso) ---")
    clean_reg_12w_df = features_df[all_predictors + ['Target_12w']].dropna()
    X_reg_12w = clean_reg_12w_df[all_predictors]
    y_reg_12w = clean_reg_12w_df['Target_12w']
    
    scaler_reg_12w = StandardScaler().fit(X_reg_12w)
    X_scaled_reg_12w = pd.DataFrame(scaler_reg_12w.transform(X_reg_12w), index=X_reg_12w.index, columns=X_reg_12w.columns)
    
    X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_scaled_reg_12w, y_reg_12w, test_size=0.2, shuffle=False)
    
    lasso_model_12w = Lasso(alpha=0.01)
    lasso_model_12w.fit(X_train_12w, y_train_12w)
    predictions_12w = lasso_model_12w.predict(X_test_12w)
    mape_12w = mean_absolute_percentage_error(y_test_12w, predictions_12w)
    print(f"✅ Final 12-Week Lasso MAPE: {mape_12w:.2%}")

    # --- 5. Save Final Models ---
    print("\n\n--- Saving final models ---")
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(classifier.fit(X_scaled_cls, y_binary), models_dir / "direction_classifier_binary_final.joblib")
    lstm_model.save(models_dir / "price_target_4w_final.h5")
    joblib.dump(lasso_model_12w.fit(X_scaled_reg_12w, y_reg_12w), models_dir / "price_target_12w_final.joblib")
    
    joblib.dump(scaler_cls, models_dir / "scaler_classifier_final.joblib")
    joblib.dump(scaler_reg_4w, models_dir / "scaler_regressor_4w_final.joblib")
    joblib.dump(scaler_reg_12w, models_dir / "scaler_regressor_12w_final.joblib")

    print("✅ All final models and scalers trained and saved successfully.")

if __name__ == "__main__":
    train_all_models()
