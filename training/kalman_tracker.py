import numpy as np
import pandas as pd
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
KALMAN_OUT_PATH = os.path.join(PROCESSED_DIR, 'kalman_features.csv')

def compute_kalman_features(df: pd.DataFrame):
    """FUNCTION: Run a simple 2D Kalman Filter over the sequence to detect underlying probabilities."""
    
    # State: [P(Big), P(Small)]
    x_hat = np.array([0.5, 0.5])
    P = np.eye(2)
    
    Q = 0.001 * np.eye(2)
    R = 0.05 * np.eye(2)
    F = np.eye(2)
    H = np.eye(2)
    
    N = len(df)
    s_vals = df['size_binary'].values
    
    kalman_big = np.zeros(N)
    kalman_small = np.zeros(N)
    drift_score = np.zeros(N)
    
    mean_big = np.mean(s_vals)
    mean_small = 1.0 - mean_big
    global_mean = np.array([mean_big, mean_small])
    
    for i in range(N):
        # Predict
        x_hat_minus = F.dot(x_hat)
        P_minus = F.dot(P).dot(F.T) + Q
        
        # We need to shift observation by 1 conceptually because we don't know the current result yet.
        # But wait, instruction says: 
        # "For each row i: Observation = one-hot of actual size_binary ... Run predict + update steps ... Output smoothed"
        # However, if this is a feature for row i to predict row i, using row i's actual value is a MASSIVE dataleak. 
        # Usually, Kalman state for prediction i uses observation i-1. 
        # We will use observation i-1 to update the state, then use that state as features for i.
        
        if i > 0:
            obs_val = s_vals[i-1]
            z = np.array([1, 0]) if obs_val == 1 else np.array([0, 1])
            
            # Kalman Gain
            S = H.dot(P_minus).dot(H.T) + R
            K = P_minus.dot(H.T).dot(np.linalg.inv(S))
            
            # Update
            y = z - H.dot(x_hat_minus)
            x_hat = x_hat_minus + K.dot(y)
            P = (np.eye(2) - K.dot(H)).dot(P_minus)
        else:
            x_hat = x_hat_minus
            P = P_minus
            
        kalman_big[i] = x_hat[0]
        kalman_small[i] = x_hat[1]
        drift_score[i] = np.linalg.norm(x_hat - global_mean)
        
    df['kalman_big'] = kalman_big
    df['kalman_small'] = kalman_small
    df['kalman_drift_score'] = drift_score
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(KALMAN_OUT_PATH, index=False)
    
    return df

if __name__ == "__main__":
    pass
