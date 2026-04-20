import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from hmmlearn import hmm

REAL_DATA_PATH = "/home/siva/Desktop/betique/okwin_30s_dataset.csv"
OUTPUT_PATH = "/home/siva/Desktop/betique/data/okwin_20k_combined.csv"
TARGET_ROWS = 20000

def generate_synthetic():
    print("--- 20k AI Smart Augmentation ---")
    if not os.path.exists(REAL_DATA_PATH):
        print("Real dataset missing.")
        return

    df_real = pd.read_csv(REAL_DATA_PATH).tail(1200) # Use recent patterns
    real_count = len(df_real)
    needed = TARGET_ROWS - real_count
    
    if needed <= 0:
        print("Target already reached.")
        return

    print(f"Seeding with {real_count} real rows. Generating {needed} synthetic rows...")
    
    # 1. Train a Hidden Markov Model on the results to capture the "vibe"
    res_stream = df_real['number'].values.reshape(-1, 1)
    
    # Simple model to learn frequencies and sequence logic
    model = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
    model.fit(res_stream)
    
    # 2. Generate new results
    synth_nums, _ = model.sample(needed)
    synth_nums = np.clip(np.round(synth_nums.flatten()), 0, 9).astype(int)
    
    # 3. Build the DataFrame
    # Work backwards from the earliest real timestamp
    try:
        first_ts = datetime.fromisoformat(df_real.iloc[0]['timestamp'])
        first_period = int(df_real.iloc[0]['period'])
    except:
        first_ts = datetime.now()
        first_period = 2026042010000000
    
    synth_list = []
    for i in range(needed):
        idx = needed - i
        ts = (first_ts - timedelta(seconds=30 * idx)).isoformat()
        pid = str(first_period - idx)
        num = int(synth_nums[i])
        size = "Big" if num >= 5 else "Small"
        colors = {0: "Red+Violet", 1: "Green", 2: "Red", 3: "Green", 4: "Red", 
                  5: "Green+Violet", 6: "Red", 7: "Green", 8: "Red", 9: "Green"}
        color_val = colors.get(num, "Unknown")
        
        synth_list.append([ts, pid, num, size, color_val])
        
    df_synth = pd.DataFrame(synth_list, columns=['timestamp', 'period', 'number', 'size', 'color'])
    
    # 4. Combine (Synthetic first, then Real historical)
    df_combined = pd.concat([df_synth, df_real], ignore_index=True)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_combined.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Result: {len(df_combined)} rows saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_synthetic()
