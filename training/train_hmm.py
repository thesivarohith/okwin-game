import pandas as pd
import numpy as np
import os
import joblib
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')
CLEAN_DATA_PATH = os.path.join(PROCESSED_DIR, 'clean_data.csv')

def hmm_predict_proba(model, sequence):
    """Predicts next emission (Small=0, Big=1) based on sequence."""
    if len(sequence) == 0:
        return np.array([0.5, 0.5])
        
    seq = np.array(sequence).reshape(-1, 1)
    try:
        # To predict next emission, find hidden state distribution after sequence
        logprob, posteriors = model.score_samples(seq)
        last_state = posteriors[-1]
        
        # Multiply by transition matrix
        next_state_dist = np.dot(last_state, model.transmat_)
        
        # Multiply by emission probability for 0 and 1
        # hmmlearn GaussianHMM defines Gaussian emissions per state via means_ and covars_
        # We cheat/simplify by evaluating the pdf for emission=0 and 1
        from scipy.stats import norm
        prob_0 = 0.0
        prob_1 = 0.0
        
        for state_idx in range(model.n_components):
            mean = model.means_[state_idx, 0]
            var = model.covars_[state_idx, 0, 0]
            std = np.sqrt(var) if var > 0 else 1e-6
            
            p_0 = norm.pdf(0, loc=mean, scale=std)
            p_1 = norm.pdf(1, loc=mean, scale=std)
            
            prob_0 += next_state_dist[state_idx] * p_0
            prob_1 += next_state_dist[state_idx] * p_1
            
        sum_prob = prob_0 + prob_1
        if sum_prob == 0:
            return np.array([0.5, 0.5])
        return np.array([prob_0 / sum_prob, prob_1 / sum_prob])
    except:
        return np.array([0.5, 0.5])

def train_hmm():
    print("\n--- Training HMM Model ---")
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    sizes = df['size_binary'].values
    
    # We are predicting position i using history [0:i]. We will simulate an expanding window or fixed window
    # Actually, for train/test probas format, we want to align with shape (N, 2) identical to others.
    # The others use WINDOW_SIZE=20 to generate (N_train) samples. We will do the same:
    # Use WINDOW_SIZE=20. Target is size_binary at step i.
    
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
    
    # Train HMM on the entire train sequence
    model = hmm.GaussianHMM(n_components=4, covariance_type='full', n_iter=200, random_state=42)
    # HMM expects (n_samples, n_features)
    train_seq = sizes[:split_idx + WINDOW_SIZE].reshape(-1, 1)
    model.fit(train_seq)
    
    # Predict probas manually
    train_probas = np.zeros((len(X_train), 2))
    for i in range(len(X_train)):
        train_probas[i] = hmm_predict_proba(model, X_train[i])
        
    test_probas = np.zeros((len(X_test), 2))
    for i in range(len(X_test)):
        test_probas[i] = hmm_predict_proba(model, X_test[i])
        
    train_preds = np.argmax(train_probas, axis=1)
    test_preds = np.argmax(test_probas, axis=1)
    
    test_acc = accuracy_score(y_test, test_preds)
    print(f"HMM Test Accuracy: {test_acc:.4f}")
    
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(SAVED_DIR, 'model_hmm.pkl'))
    
    np.save(os.path.join(SAVED_DIR, 'hmm_train_proba.npy'), train_probas)
    np.save(os.path.join(SAVED_DIR, 'hmm_test_proba.npy'), test_probas)
    
    return model, test_acc

if __name__ == "__main__":
    pass
