import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')
CLEAN_DATA_PATH = os.path.join(PROCESSED_DIR, 'clean_data.csv')

def train_bilstm():
    print("\n--- Training Bi-LSTM Model ---")
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    WINDOW_SIZE = 20
    
    # Normalize features
    results_norm = df['result'].values / 9.0
    sizes = df['size_binary'].values
    colors_norm = df['color_enc'].values / 2.0  # Actually max is 2 for color_enc
    
    # Input shape: (N, 20, 3)
    X = []
    y = []
    
    # Create sliding window sequences
    for i in range(WINDOW_SIZE, len(df)):
        seq_res = results_norm[i - WINDOW_SIZE : i]
        seq_size = sizes[i - WINDOW_SIZE : i]
        seq_col = colors_norm[i - WINDOW_SIZE : i]
        
        sequence = np.column_stack([seq_res, seq_size, seq_col])
        X.append(sequence)
        
        # Target is size_binary at step i
        if sizes[i] == 1:
            y.append([0, 1]) # [Small, Big]
        else:
            y.append([1, 0])
            
    X = np.array(X)
    y = np.array(y)
    
    # Split 70/30 (NO SHUFFLE)
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(WINDOW_SIZE, 3)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )
    
    loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Bi-LSTM Test Accuracy: {test_acc:.4f}")
    
    os.makedirs(SAVED_DIR, exist_ok=True)
    model.save(os.path.join(SAVED_DIR, 'model_lstm.keras'))
    
    train_probas = model.predict(X_train, verbose=0)
    test_probas = model.predict(X_test, verbose=0)
    
    np.save(os.path.join(SAVED_DIR, 'lstm_train_proba.npy'), train_probas)
    np.save(os.path.join(SAVED_DIR, 'lstm_test_proba.npy'), test_probas)
    
    return model, test_acc

if __name__ == "__main__":
    pass
