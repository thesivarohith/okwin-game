"""
Step 5: Build Higher-Order Markov Chain for Big/Small transitions.
Input:  data/clean_data.csv
Output: model/markov_model.pkl
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
from collections import defaultdict

# --- Paths ---
CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'markov_model.pkl')

MAX_ORDER = 5
STATES = [0, 1]  # Small=0, Big=1

# Import from shared module
from markov_model import MarkovChain


def main():
    if not os.path.exists(CLEAN_PATH):
        print(f"ERROR: {CLEAN_PATH} not found. Run preprocess.py first.")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_csv(CLEAN_PATH)
    sequence = df['target'].values.tolist()
    print(f"  Sequence length: {len(sequence)}")

    # Split (70/30 time-series)
    split_idx = int(len(sequence) * 0.7)
    train_seq = sequence[:split_idx]
    test_seq = sequence[split_idx:]
    print(f"  Train: {len(train_seq)}, Test: {len(test_seq)}")

    # Fit
    print(f"\nFitting Markov Chain (orders 1-{MAX_ORDER})...")
    mc = MarkovChain(max_order=MAX_ORDER)
    mc.fit(train_seq)

    # Print transition matrices
    for order in range(1, MAX_ORDER + 1):
        n_states = len(mc.transition_probs.get(order, {}))
        print(f"  Order {order}: {n_states} unique state patterns")

    # Evaluate on test set
    print("\nEvaluating...")
    correct = 0
    total = 0
    for i in range(MAX_ORDER, len(test_seq)):
        # Use full history up to this point for prediction
        history = sequence[:split_idx + i]
        pred = mc.predict(history)
        actual = test_seq[i]
        if pred == actual:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"  Test Accuracy: {acc:.4f} ({correct}/{total})")

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(mc, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")

    return acc


if __name__ == '__main__':
    main()
