"""
Step 6: Train Bidirectional LSTM for Big/Small prediction.
Input:  data/clean_data.csv
Output: model/model_lstm.keras
"""
import pandas as pd
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_lstm.keras')

WINDOW_SIZE = 20  # sliding window of last 20 rounds
FEATURES_PER_STEP = 3  # result, target (size), color_encoded


def create_sequences(df, window_size=WINDOW_SIZE):
    """Create sliding window sequences for LSTM."""
    results = df['result'].values.astype(float)
    targets = df['target'].values.astype(float)
    colors = df['color_encoded'].values.astype(float)

    # Normalize result to 0-1
    results_norm = results / 9.0

    X, y = [], []
    for i in range(window_size, len(df)):
        seq = np.column_stack([
            results_norm[i - window_size:i],
            targets[i - window_size:i],
            colors[i - window_size:i] / 4.0,  # normalize color
        ])
        X.append(seq)
        y.append(targets[i])

    return np.array(X), np.array(y)


def build_model(window_size, n_features):
    """Build Bi-LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Bidirectional, LSTM, Dense, Dropout, BatchNormalization
    )

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(window_size, n_features)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    if not os.path.exists(CLEAN_PATH):
        print(f"ERROR: {CLEAN_PATH} not found. Run preprocess.py first.")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_csv(CLEAN_PATH)
    print(f"  Rows: {len(df)}")

    # Create sequences
    print(f"Creating sequences (window={WINDOW_SIZE})...")
    X, y = create_sequences(df, WINDOW_SIZE)
    print(f"  Sequences: {X.shape[0]}, Window: {X.shape[1]}, Features: {X.shape[2]}")

    # Time-series split (70/30, no shuffle)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Build and train
    print("\nTraining Bi-LSTM...")
    model = build_model(WINDOW_SIZE, FEATURES_PER_STEP)
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss'),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Small', 'Big']))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")

    return acc


if __name__ == '__main__':
    main()
