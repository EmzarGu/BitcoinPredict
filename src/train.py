import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge # <-- This is the fix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import build_features, PREDICTOR_COLS

def create_sequences(X, y, time_steps=4):
    """Create sequences for LSTM model."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_all_models():
    """
    Trains, evaluates, and saves all predictive models.
    Now includes an LSTM model for 4-week price forecasting.
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

    # --- 3. Split Data ---
    X_train_4w, X_test_4w, y_train_4w, y_test_4w = train_test_split(X_4w_data, y_4w, test_size=0.2, shuffle=False)
    X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_12w_data, y_12w, test_size=0.2, shuffle=False)
    
    # --- 4. Train Directional Classifier (as before) ---
    y_train_4w_class = y_train_4w.apply(lambda ret: 2 if ret > 0.05 else (0 if ret < -0.05 else 1))
    y_test_4w_class = y_test_4w.apply(lambda ret: 2 if ret > 0.05 else (0 if ret < -0.05 else 1))
    
    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss',
                                   gamma=0, learning_rate=0.01, max_depth=3, n_estimators=100)
    classifier.fit(X_train_4w, y_train_4w_class)
    accuracy = accuracy_score(y_test_4w_class, classifier.predict(X_test_4w))
    print(f"\n✅ Directional Classifier (4-Week):")
    print(f"   - Test Accuracy: {accuracy:.2%}")

    # --- 5. Train Price Target Regressors ---
    
    # 4-Week Regressor (LSTM)
    print("\n--- Training Price Target Regressor (4-Week, LSTM)... ---")
    scaler_X = MinMaxScaler()
    X_train_4w_scaled = scaler_X.fit_transform(X_train_4w)
    X_test_4w_scaled = scaler_X.transform(X_test_4w)
    
    time_steps = 4
    X_train_seq, y_train_seq = create_sequences(pd.DataFrame(X_train_4w_scaled), y_train_4w, time_steps)
    X_test_seq, y_test_seq = create_sequences(pd.DataFrame(X_test_4w_scaled), y_test_4w, time_steps)
    
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
                   validation_data=(X_test_seq, y_test_seq), callbacks=[early_stopping], verbose=1)
    
    y_pred_lstm = lstm_model.predict(X_test_seq)
    mape_4w_lstm = mean_absolute_percentage_error(y_test_seq, y_pred_lstm)
    print(f"\n✅ Price Target Regressor (4-Week, LSTM):")
    print(f"   - Test MAPE: {mape_4w_lstm:.2%}")

    # 12-Week Regressor (Bayesian Ridge)
    bayesian_12w = BayesianRidge()
    bayesian_12w.fit(X_train_12w, y_train_12w)
    mape_12w_bayes = mean_absolute_percentage_error(y_test_12w, bayesian_12w.predict(X_test_12w))
    print(f"\n✅ Price Target Regressor (12-Week, BayesianRidge):")
    print(f"   - Test MAPE: {mape_12w_bayes:.2%}")

    # --- 6. Re-train Final Models on All Data and Save ---
    print("\n\n--- Re-training final models on all available data ---")
    full_y_4w_class = y_4w.apply(lambda ret: 2 if ret > 0.05 else (0 if ret < -0.05 else 1))
    classifier.fit(X_4w_data, full_y_4w_class)
    
    X_full_4w_scaled = scaler_X.fit_transform(X_4w_data)
    X_full_seq, y_full_seq = create_sequences(pd.DataFrame(X_full_4w_scaled), y_4w, time_steps)
    lstm_model.fit(X_full_seq, y_full_seq, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    
    bayesian_12w.fit(X_12w_data, y_12w)

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, models_dir / "direction_classifier_xgb.joblib")
    joblib.dump(scaler_X, models_dir / "scaler_X_4w.joblib") # Save the scaler
    lstm_model.save(models_dir / "price_target_4w_lstm.h5") # Save the LSTM model
    joblib.dump(bayesian_12w, models_dir / "price_target_12w_bayes.joblib")

    print("✅ All final models trained and saved successfully.")

if __name__ == "__main__":
    train_all_models()
