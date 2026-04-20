import pandas as pd
import numpy as np
import os
import joblib
from collections import defaultdict
from sklearn.metrics import accuracy_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')
CLEAN_DATA_PATH = os.path.join(PROCESSED_DIR, 'clean_data.csv')

def predict_markov(matrices, recent_seq):
    """Predicts using highest possible order with >=10 occurrences."""
    q_len = len(recent_seq)
    for order in sorted(matrices.keys(), reverse=True):
        if q_len >= order:
            state = tuple(recent_seq[-order:])
            if state in matrices[order]:
                counts = matrices[order][state]
                total = sum(counts.values())
                if total >= 10:
                    prob_small = counts.get(0, 0) / total
                    prob_big = counts.get(1, 0) / total
                    return np.array([prob_small, prob_big])
    return np.array([0.5, 0.5])

def train_markov():
    print("\n--- Training Markov Chain Model ---")
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    sizes = df['size_binary'].values
    
    WINDOW_SIZE = 20
    X = []
    y = []
    
    for i in range(WINDOW_SIZE, len(df)):
        X.append(sizes[i - WINDOW_SIZE : i])
        y.append(sizes[i])
        
    X = np.array(X)
    y = np.array(y)
    
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Train matrices on train set
    matrices = {}
    train_seq = sizes[:split_idx + WINDOW_SIZE]
    
    for order in [1, 2, 3, 4, 5]:
        mat = defaultdict(lambda: {0: 0, 1: 0})
        for i in range(len(train_seq) - order):
            state = tuple(train_seq[i : i + order])
            next_val = train_seq[i + order]
            mat[state][next_val] += 1
        matrices[order] = dict(mat)
        
    # Generate probas
    train_probas = np.zeros((len(X_train), 2))
    for i in range(len(X_train)):
        train_probas[i] = predict_markov(matrices, X_train[i])
        
    test_probas = np.zeros((len(X_test), 2))
    for i in range(len(X_test)):
        test_probas[i] = predict_markov(matrices, X_test[i])
        
    train_preds = np.argmax(train_probas, axis=1)
    test_preds = np.argmax(test_probas, axis=1)
    
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Markov Test Accuracy: {test_acc:.4f}")
    
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(matrices, os.path.join(SAVED_DIR, 'markov_model.pkl'))
    
    np.save(os.path.join(SAVED_DIR, 'markov_train_proba.npy'), train_probas)
    np.save(os.path.join(SAVED_DIR, 'markov_test_proba.npy'), test_probas)
    
    return matrices, test_acc

if __name__ == "__main__":
    pass
