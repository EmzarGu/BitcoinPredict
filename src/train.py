import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    Trains the final, most accurate system:
    - Classifier: Binary Logistic Regression on high-importance features.
    - Regressor: LSTM for price targets.
    """
    print("--- Starting Final Model Training ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Define Feature Set and Prepare Data ---
    # Use the high-importance features that gave us our best result
    high_importance_features = [
        'Liquidity_Z', 'LGC_distance_z',
        'SMA_ratio_52w', 'Realised_to_Spot'
    ]
    
    # Also include all predictors needed for the LSTM
    PREDICTOR_COLS = list(set(features_df.columns) - {'Target', 'Target_12w', 'close_usd'})


    clean_df = features_df[PREDICTOR_COLS + ['Target']].dropna()
    
    # --- 3. Train Binary Directional Classifier ---
    print("\n--- Training Binary Directional Classifier ---")
    
    X_class = clean_df[high_importance_features]
    y_binary = (clean_df['Target'] > 0.01).astype(int) # 1 for UP, 0 for NOT UP
    
    scaler_cls = StandardScaler()
    X_scaled_cls = pd.DataFrame(scaler_cls.fit_transform(X_class), index=X_class.index, columns=X_class.columns)

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled_cls, y_binary, test_size=0.2, shuffle=False)
    
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_cls, y_train_cls)
    accuracy = accuracy_score(y_test_cls, classifier.predict(X_test_cls))
    print(f"✅ Final Binary Classifier Accuracy: {accuracy:.2%}")

    # --- 4. Train Price Target Regressor (LSTM) ---
    print("\n--- Training Price Target Regressor (LSTM) ---")
    X_reg = clean_df[PREDICTOR_COLS]
    y_reg = clean_df['Target']
    
    scaler_reg = StandardScaler()
    X_scaled_reg = pd.DataFrame(scaler_reg.fit_transform(X_reg), index=X_reg.index, columns=X_reg.columns)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled_reg, y_reg, test_size=0.2, shuffle=False)

    time_steps = 4
    X_train_seq, y_train_seq = create_sequences(X_train_reg, y_train_reg, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_reg, y_test_reg, time_steps)

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

    predictions = lstm_model.predict(X_test_seq)
    mape = mean_absolute_percentage_error(y_test_seq, predictions)
    print(f"✅ Final LSTM MAPE: {mape:.2%}")

    # --- 5. Save the Final Models ---
    print("\n\n--- Saving the best performing models ---")
    classifier.fit(X_scaled_cls, y_binary)
    
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, models_dir / "direction_classifier_binary_final.joblib")
    lstm_model.save(models_dir / "price_target_regressor_final.h5")
    joblib.dump(scaler_cls, models_dir / "scaler_classifier_final.joblib")
    joblib.dump(scaler_reg, models_dir / "scaler_regressor_final.joblib")

    print("✅ All final models trained and saved successfully.")

if __name__ == "__main__":
    train_all_models()
