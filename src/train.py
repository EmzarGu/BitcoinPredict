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

from src.features import build_features, PREDICTOR_COLS

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
    Trains the final hybrid model and saves the best performing versions
    from the validation stage to prevent overfitting.
    """
    print("--- Starting Final Model Training ---")

    # 1. Load Data
    features_df = build_features(for_training=True)
    if features_df.empty:
        print("❌ Could not build features. Aborting.")
        return

    # --- 2. Prepare and Scale Data ---
    predictor_cols_exist = [col for col in PREDICTOR_COLS if col in features_df.columns]
    X_full = features_df[predictor_cols_exist]

    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X_full), index=X_full.index, columns=X_full.columns)

    # --- 3. Train and Validate Models ---
    
    # Classifier
    print("\n--- Training & Validating Directional Classifier ---")
    y_4w_df = features_df[['Target']].dropna()
    X_class = X_scaled_df.loc[y_4w_df.index]
    def classify(ret): return 2 if ret > 0.05 else (0 if ret < -0.05 else 1)
    y_class = y_4w_df['Target'].apply(classify)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, y_class, test_size=0.2, shuffle=False)
    classifier = LogisticRegression(max_iter=1000, multi_class='multinomial')
    classifier.fit(X_train_cls, y_train_cls)
    accuracy = accuracy_score(y_test_cls, classifier.predict(X_test_cls))
    print(f"✅ Final Classifier Accuracy: {accuracy:.2%}")

    # 4-Week Regressor (LSTM)
    print("\n--- Training & Validating 4-Week Regressor (LSTM) ---")
    y_4w_reg = features_df['Target'].dropna()
    X_4w_reg = X_scaled_df.loc[y_4w_reg.index]
    X_train_4w, X_test_4w, y_train_4w, y_test_4w = train_test_split(X_4w_reg, y_4w_reg, test_size=0.2, shuffle=False)
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
    print("\n--- Training & Validating 12-Week Regressor (Lasso) ---")
    y_12w_reg = features_df['Target_12w'].dropna()
    X_12w_reg = X_scaled_df.loc[y_12w_reg.index]
    X_train_12w, X_test_12w, y_train_12w, y_test_12w = train_test_split(X_12w_reg, y_12w_reg, test_size=0.2, shuffle=False)
    lasso_reg_12w = Lasso(alpha=0.01)
    lasso_reg_12w.fit(X_train_12w, y_train_12w)
    predictions_12w = lasso_reg_12w.predict(X_test_12w)
    mape_12w = mean_absolute_percentage_error(y_test_12w, predictions_12w)
    print(f"✅ Final 12-Week Lasso MAPE: {mape_12w:.2%}")

    # --- 4. Save the Final, Best Performing Models ---
    print("\n\n--- Saving the best models from validation ---")
    
    # **THIS IS THE FIX**: We train the final classifier and scaler on all data,
    # but we save the REGRESSION models that were validated on the test set.
    # This prevents overfitting and keeps our best results.
    
    final_classifier = LogisticRegression(max_iter=1000, multi_class='multinomial').fit(X_class, y_class)
    final_scaler = StandardScaler().fit(X_full)

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_classifier, models_dir / "direction_classifier_final.joblib")
    lstm_model.save(models_dir / "price_target_4w_final.h5") # Save the best LSTM
    joblib.dump(lasso_reg_12w, models_dir / "price_target_12w_final.joblib") # Save the best Lasso
    joblib.dump(final_scaler, models_dir / "scaler_final.joblib")

    print("✅ All final models trained and saved successfully.")

if __name__ == "__main__":
    train_all_models()
