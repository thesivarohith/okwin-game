"""
MarkovChain class — shared module used by training and backend.
"""
import numpy as np
from collections import defaultdict

STATES = [0, 1]  # Small=0, Big=1


class MarkovChain:
    """Higher-order Markov Chain for Big/Small prediction."""

    def __init__(self, max_order=5):
        self.max_order = max_order
        self.transition_counts = {}
        self.transition_probs = {}

    def fit(self, sequence):
        """Build transition matrices from sequence."""
        for order in range(1, self.max_order + 1):
            counts = defaultdict(lambda: defaultdict(int))
            for i in range(order, len(sequence)):
                state = tuple(sequence[i - order:i])
                next_state = sequence[i]
                counts[state][next_state] += 1
            self.transition_counts[order] = dict(counts)

            probs = {}
            for state, next_counts in counts.items():
                total = sum(next_counts.values())
                probs[state] = {s: next_counts.get(s, 0) / total for s in STATES}
            self.transition_probs[order] = probs

    def predict_proba(self, history):
        """Given recent history, predict P(Small) and P(Big)."""
        combined_prob = np.array([0.5, 0.5])
        weight_sum = 0

        for order in range(self.max_order, 0, -1):
            if len(history) < order:
                continue
            state = tuple(history[-order:])
            if order in self.transition_probs and state in self.transition_probs[order]:
                probs = self.transition_probs[order][state]
                p = np.array([probs.get(0, 0.5), probs.get(1, 0.5)])
                w = order
                combined_prob = combined_prob + w * p
                weight_sum += w

        if weight_sum > 0:
            combined_prob = combined_prob / (1 + weight_sum)

        combined_prob = combined_prob / combined_prob.sum()
        return combined_prob

    def predict(self, history):
        """Return predicted class (0=Small, 1=Big)."""
        proba = self.predict_proba(history)
        return int(np.argmax(proba))
