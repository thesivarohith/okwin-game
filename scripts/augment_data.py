import pandas as pd
import numpy as np
import os

def augment_data():
    raw_path = "data/raw_data.csv"
    if not os.path.exists(raw_path):
        print("Error: data/raw_data.csv not found.")
        return

    df = pd.read_csv(raw_path)
    current_count = len(df)
    target = 15000
    needed = target - current_count
    
    if needed <= 0:
        df.to_csv("data/augmented_15k.csv", index=False)
        return

    print(f"Augmenting {current_count} real records with {needed} synthetic ones...")
    
    # Simple Markov generation
    df['size_binary'] = (df['size'] == 'Big').astype(int)
    seq = df['size_binary'].iloc[::-1].values
    matrix = np.zeros((2, 2))
    for i in range(len(seq) - 1):
        matrix[seq[i], seq[i+1]] += 1
    transition_probs = matrix / matrix.sum(axis=1)[:, np.newaxis]
    
    num_counts = df['result'].value_counts(normalize=True).sort_index()
    
    synth_results = []
    current_state = seq[-1]
    for i in range(needed):
        next_state = np.random.choice([0, 1], p=transition_probs[current_state])
        nums = [n for n in range(10) if (n >= 5 if next_state == 1 else n < 5)]
        p = [num_counts.get(n, 0.1) for n in nums]
        p = np.array(p) / sum(p)
        chosen_num = np.random.choice(nums, p=p)
        
        synth_results.append({
            "timestamp": "SYNTH",
            "period": f"SYNTH_{i}",
            "result": int(chosen_num),
            "size": "Big" if next_state == 1 else "Small",
            "color": "Green" if chosen_num % 2 != 0 else "Red"
        })
        current_state = next_state
        
    final_df = pd.concat([df, pd.DataFrame(synth_results)])
    final_df.to_csv("data/augmented_15k.csv", index=False)
    print("Saved to data/augmented_15k.csv")

if __name__ == "__main__":
    augment_data()
